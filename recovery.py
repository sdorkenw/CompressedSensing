from matplotlib import pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import os
from imageio import imread, imwrite
import scipy.sparse
import sklearn.linear_model as lm
import scipy.ndimage
import sys

import plotting
import utils


def sampling_mask(shape, rate=0.5):
    """ Creates sampling mask

    :param shape: (int, int)
        shape of mask
    :param rate: float
        ratio of True pixels
    :return: 2d bool array
        mask
    """
    assert 0 <= rate < 1.

    np.random.seed( 42 ) # !!!La risposta ad ogni domanda
    size = int(np.product( shape ) )
    mask = np.zeros( size, dtype = bool )
    mask[np.random.choice(size, int(size*rate), replace=False)] = True
    return mask.reshape(shape)


def recovery_single(img, mask, basis="dct", wvt_level=3, alpha=None):
    """ Recovers image

    :param img: 2d array
        corrupted image
    :param mask: 2d bool array
        mask
    :param basis: str
        either 'wvt', 'dct'
    :param wvt_level: int
        level of wavelet transform
    :param alpha: float or None
        regularization parameter
        if None: alpha is first found with CV (takes long)
    :return: 2d array
        recovered image
    """
    print ("img shape: ", img.shape)
    img_f = img.flatten()
    A = np.eye(len(img_f), dtype=bool)
    print("Build sensing matrix")
    mask_f = mask.flatten()
    for i_px in range(len(img_f)):
        if mask_f[i_px]:
            A[i_px, i_px] = False

    print("Measurement")
    b = np.array([np.dot(A[i], img_f) for i in range(len(img_f))])

    if basis == "wvt":
        print("Wavelets")
        trans_A = [utils.dwt2(A[i].reshape(img.shape), level=wvt_level).
                     astype(np.float16).flatten() for i in range(len(img_f))]
    elif basis == "dct":
        print("DCT")
        trans_A = [utils.dct2(A[i].reshape(img.shape)).
                       astype(np.float16).flatten() for i in range(len(img_f))]
    else:
        raise Exception("Unknown basis")

    A = None

    if alpha:
        lasso = lm.Lasso(alpha=alpha, max_iter=100000, normalize=True)
        print("Fit")
        lasso.fit(trans_A, b)
    else:
        lasso_cv = lm.LassoCV(n_jobs=cpu_count(), max_iter=100000,
                              normalize=True)
        print("Fit")
        lasso_cv.fit(trans_A, b)
        print("Alpha: %.6f" % lasso_cv.alpha_)
        lasso = lm.Lasso(alpha=lasso_cv.alpha_, max_iter=100000,
                         normalize=True)
        print("Fit")
        lasso.fit(trans_A, b)

    if basis == "wvt":
        return utils.idwt2(lasso.coef_.reshape(img.shape), level=wvt_level)
    elif basis == "dct":
        return utils.idct2(lasso.coef_.reshape(img.shape))
    else:
        raise Exception("Unknown basis")


def _recovery_thread(args):
    rec = recovery_single(args[1], args[2], wvt_level=args[3], alpha=args[4])
    return args[0], rec


def recover_blocks(img, mask, blocksize=32, wvt_level=3, alpha=None,
                   n_processes=None):
    """ Recovers image parallel with blocking

    :param img: 2d array
        image
    :param mask: 2d bool array
        mask
    :param blocksize: int
        defines edgelength of blocking squares
    :param wvt_level: int
        level of wavelet transform
    :param alpha: float or None
        regularization parameter
        if None: alpha is first found with CV (takes long)
    :param n_processes: int or None
        number of processes to be used
        if None: all cores are used
    :return: 2d array
        recovered image
    """
    if not n_processes:
        n_processes = cpu_count()

    my_img = np.pad(img, [[blocksize//2, blocksize//2],
                          [blocksize//2, blocksize//2]], mode="reflect")
    my_mask = np.pad(mask, [[blocksize//2, blocksize//2],
                            [blocksize//2, blocksize//2]], mode="reflect")

    params = []

    for x_pos in range(0, my_img.shape[0]-blocksize//2, blocksize//2):
        for y_pos in range(0, my_img.shape[1]-blocksize//2, blocksize//2):
            params.append([[x_pos, y_pos],
                           my_img[x_pos: x_pos + blocksize,
                                  y_pos: y_pos + blocksize],
                           my_mask[x_pos: x_pos + blocksize,
                                   y_pos: y_pos + blocksize],
                           wvt_level, alpha, blocksize])

    print("N jobs: %d" % (len(params)))
    if n_processes > 1:
        pool = Pool(n_processes)
        results = pool.map(_recovery_thread, params)
        pool.close()
        pool.join()
    else:
        results = map(_recovery_thread, params)

    r_img = np.zeros(my_img.shape, dtype=np.float32)
    normalization = np.zeros(my_img.shape, dtype=np.float16)
    for result in results:
        r_img[result[0][0] + blocksize//4: result[0][0] + blocksize*3//4,
              result[0][1] + blocksize//4: result[0][1] + blocksize*3//4] += \
            result[1][blocksize//4: blocksize*3//4, blocksize//4:blocksize*3//4]
        normalization[result[0][0] + blocksize//4:
                                            result[0][0] + blocksize*3//4,
                      result[0][1] + blocksize//4:
                                            result[0][1] + blocksize*3//4] += 1

    r_img /= normalization
    return r_img[blocksize//2: -blocksize//2, blocksize//2: -blocksize//2]


def recover_main(img_path, zoom_rate=1., corruption_rate=.1, alpha=None,
                 wvt_level=3, save_path=None, blocksize=32, n_processes=None):
    """ Recoveres an image and performs assisting work

    :param img_path: str
        path to true image
    :param zoom_rate: float
        image rescaling factor
    :param corruption_rate: float
        corruptin rate (0-1)
    :param alpha: float or None
        regularization parameter
        if None: alpha is first found with CV (takes long)
    :param wvt_level: int
        level of wavelet transform
    :param save_path: str
        path to folder to which images will be exported
        if ends with .png: only recovered image and cropped image are stored
        if None: images will be shown
    :param blocksize: int
        defines edgelength of blocking squares
    :param n_processes:int or None
        number of processes to be used
        if None: all cores are used
    """

    assert os.path.exists(img_path)
    img = imread(img_path)

    if len(img.shape) > 2:
        print("Only grayscale supported...")
        img = np.mean(img, axis=-1)
        print("... RGB converted to grayscale")

    if zoom_rate != 1:
        img = scipy.ndimage.zoom(img, zoom=zoom_rate, order=3)

    off = [(img.shape[0] % 2 ** wvt_level) / 2.,
           (img.shape[1] % 2 ** wvt_level) / 2.]

    img = img[int(np.floor(off[0])): img.shape[0] - int(np.ceil(off[0])),
              int(np.floor(off[1])): img.shape[1] - int(np.ceil(off[1]))]

    if (3 + wvt_level)**2 > blocksize:
        print ("Blocksize is too small for chosen wavelet level...")
        wvt_level = int(np.log2(blocksize)) - 3
        print ( "... chose %d as wavelet level instead." % wvt_level)

    print("Cropped image to: ", img.shape)

    mask = sampling_mask(img.shape, rate=corruption_rate)

    img = img.astype(np.float16) / np.max(img)

    d_img = img.copy()
    d_img[mask] = 0

    r_img = recover_blocks(d_img.copy(), mask.copy(), alpha=alpha,
                           blocksize=blocksize, n_processes=n_processes,
                           wvt_level=wvt_level)
    r_img[r_img < 0] = 0
    neg_mask = np.invert(mask)

    print("rmse %.3f" % (utils.compute_rmse(img, r_img)))
    print("Corruption rate: %.3f, number of pixel: %d" % (corruption_rate,
          int(np.sum(neg_mask))))

    if save_path:
        if save_path.endswith(".png"):
            imwrite(save_path, r_img)
            imwrite(save_path[:-4] + "_true.png", d_img)
        else:
            if not os.path.exists(save_path + "/cr_%d/" % (corruption_rate * 1000)):
                os.makedirs(save_path + "/cr_%d/" % (corruption_rate * 1000))
            alpha_s = int(alpha * 1e9)
            imwrite(save_path + "/cr_%d/img_block_size_block_size_%d_lambda_%dx1e-9.png" %
                   (corruption_rate * 1000, blocksize, alpha_s), img)
            imwrite(save_path + "/cr_%d/corrupt_img_block_size_%d_lambda_%dx1e-9.png" %
                   (corruption_rate * 1000, blocksize, alpha_s), d_img)
            imwrite(save_path + "/cr_%d/rec_%d_%d.png" %
                   (corruption_rate * 1000, blocksize, alpha_s), r_img)
            imwrite(save_path + "/cr_%d/rec_diff_block_size_%d_lambda_%dx1e-9.png" %
                   (corruption_rate * 1000, blocksize, alpha_s),
                   np.abs(img-r_img))
            imwrite(save_path + "/cr_%d/rec_diff_block_size_%d_lambda_%dx1e-9_gray.png" %
                   (corruption_rate * 1000, blocksize, alpha_s), img-r_img)
            plotting.plot_img(img-r_img,
                              save_path + "/cr_%d/rec_diff_block_size_%d_lambda_%dx1e-9.png" %
                              (corruption_rate * 1000, blocksize, alpha_s))
    else:
        plt.clf()
        fig, axarr = plt.subplots(2, 2)

        axarr[0, 0].imshow(img, cmap="gray")
        axarr[0, 1].imshow(d_img, cmap="gray")
        axarr[1, 0].imshow(r_img, cmap="gray")
        axarr[1, 1].imshow(np.abs(r_img-img), cmap="gray")

        plt.show()

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage for parameter swiping: python2 recovery.py <path_to_img> "
              "<path_to_save_folder> [<blocksize>]")
    else:
        assert os.path.exists(sys.argv[1])
        img_path = sys.argv[1]

        assert os.path.exists(sys.argv[2])
        save_path = sys.argv[2]

        if len(sys.argv) == 4:
            blocksizes = [int(sys.argv[3])]
        else:
            blocksizes = [16, 32, 64]

        alphas = list(np.logspace(-7, 1, num=20))
        corruption_rates = [i / 10. for i in range(1, 10)] + [.95, .99]

        # for cr in corruption_rates:
        #     for b in blocksizes:
        #         for a in alphas:
        #             recover_main(img_path=img_path, corruption_rate=cr, alpha=a,
        #                          blocksize=b, save_path=save_path)

        recover_main(img_path=img_path, corruption_rate=corruption_rates[3], alpha=alphas[3],
                        blocksize=blocksizes[0], save_path=save_path)

        utils.compute_rmse_folder(save_path + "/cr_%d" % (corruption_rates[3]*1000))
