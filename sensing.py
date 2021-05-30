from matplotlib import pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
from imageio import imread, imwrite
import scipy.ndimage
import sklearn.linear_model as lm
import sys
import os

import utils


def sense(img, k=1000, basis="wvt", wvt_level=4, alpha=None):
    """ Senses an image with k measurements

    :param img: 2d array
        image
    :param k: int
        number of measurements
    :param basis: str
        either 'wvt', 'dct'
    :param wvt_level: int
        level of wavelet transform
    :param alpha: float or None
        regularization parameter
        if None: alpha is first found with CV (takes long)
    :return: 2d array
        sensed image
    """
    print ("Image size:", img.shape )
    img_f = img.flatten()
    print ("Build sensing matrix")

    A = np.random.normal(0, 1, len(img_f)*k).astype(np.float32).\
        reshape(k, img.shape[0], img.shape[1])

    print ("Measurement")
    b = np.dot(A.reshape(k, len(img_f)), img_f)

    if basis == "wvt":
        print ("Wavelets")
        trans_A = [utils.dwt2(A[i].reshape(img.shape), level=wvt_level).
                     astype(np.float16).flatten() for i in range(k)]
    elif basis == "dct":
        print ("DCT" )
        trans_A = [utils.dct2(A[i].reshape(img.shape)).
                       astype(np.float16).flatten() for i in range(k)]
    else:
        raise Exception("Unknown basis")

    A = None

    if alpha:
        lasso = lm.Lasso(alpha=alpha, max_iter=100000, normalize=True)
        print ( "Fit" )
        lasso.fit(trans_A, b)
    else:
        lasso_cv = lm.LassoCV(n_jobs=cpu_count(), max_iter=100000,
                              normalize=True)
        print ("Fit")
        lasso_cv.fit(trans_A, b)
        print ("Alpha: %.6f" % lasso_cv.alpha_)
        lasso = lm.Lasso(alpha=lasso_cv.alpha_, max_iter=100000,
                         normalize=True)
        print ("Fit")
        lasso.fit(trans_A, b)

    if basis == "wvt":
        return utils.idwt2(lasso.coef_.reshape(img.shape), level=wvt_level)
    elif basis == "dct":
        return utils.idct2(lasso.coef_.reshape(img.shape))
    else:
        raise Exception("Unknown basis")


def _sense_block_thread(args):
    rec = sense(args[1], args[2], wvt_level=args[3], basis=args[4],
                alpha=args[5])
    return args[0], rec


def sense_blocks(img, ratio, blocksize=64, wvt_level=3, alpha=None,
                 basis="wvt", n_processes=None):
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

    my_img = np.pad(img, [[blocksize/2, blocksize/2],
                          [blocksize/2, blocksize/2]], mode="reflect")

    k = int((blocksize/2)**2 * ratio)

    params = []

    for x_pos in range(0, my_img.shape[0]-blocksize/2, blocksize/2):
        for y_pos in range(0, my_img.shape[1]-blocksize/2, blocksize/2):
            params.append([[x_pos, y_pos],
                           my_img[x_pos: x_pos + blocksize,
                                  y_pos: y_pos + blocksize],
                           k, wvt_level, basis, alpha])

    print("N jobs: %d" % (len(params)))
    if n_processes > 1:
        pool = Pool(n_processes)
        results = pool.map(_sense_block_thread, params)
        pool.close()
        pool.join()
    else:
        results = map(_sense_block_thread, params)

    r_img = np.zeros(my_img.shape, dtype=np.float32)
    normalization = np.zeros(my_img.shape, dtype=np.float16)
    for result in results:
        r_img[result[0][0] + blocksize/4: result[0][0] + blocksize*3/4,
              result[0][1] + blocksize/4: result[0][1] + blocksize*3/4] += \
            result[1][blocksize/4: blocksize*3/4, blocksize/4:blocksize*3/4]
        normalization[result[0][0] + blocksize/4:
                                            result[0][0] + blocksize*3/4,
                      result[0][1] + blocksize/4:
                                            result[0][1] + blocksize*3/4] += 1

    r_img /= normalization
    return r_img[blocksize/2: -blocksize/2, blocksize/2: -blocksize/2]


def sense_main(img_path, zoom_rate=1., k=1000, alpha=None, wvt_level=5,
               add_noise=True, basis="dct", save_path=None):
    """ Senses an image with k measurements and performs assisting work

    :param img_path: str
        path to true image
    :param zoom_rate: float
        image rescaling factor
    :param k: int
        number of measurements / height of sensing matrix
    :param alpha: float or None
        regularization parameter
        if None: alpha is first found with CV (takes long)
    :param wvt_level: int
        level of wavelet transform
    :param add_noise: bool
        adds random noise to the image to reduce SNR
    :param basis: str
        either 'wvt', 'dct'
    :param save_path: str
        path to folder to which images will be exported
        if ends with .png: only recovered image and cropped image are stored
        if None: images will be shown
    """
    assert os.path.exists(img_path)
    img = imread(img_path)

    if len(img.shape) > 2:
        print ("Only grayscale supported...")
        img = np.mean(img, axis=-1)
        print ("... RGB converted to grayscale")

    if zoom_rate != 1:
        img = scipy.ndimage.zoom(img, zoom=zoom_rate, order=3)

    if basis == "wvt":
        off = [(img.shape[0] % 2 ** wvt_level) / 2.,
               (img.shape[1] % 2 ** wvt_level) / 2.]

        img = img[int(np.floor(off[0])): img.shape[0] - int(np.ceil(off[0])),
                  int(np.floor(off[1])): img.shape[1] - int(np.ceil(off[1]))]

        print ("Cropped image to: ", img.shape)

        img = img.astype(np.float16) / np.max(img)

    if add_noise:
        nsy_img = utils.decrease_SNR(img)
    else:
        nsy_img = img

    if True:
        r_img = sense(nsy_img.copy(), k=k, alpha=alpha, wvt_level=wvt_level,
                      basis=basis)
    else:
        r_img = sense_blocks(nsy_img.copy(),
                             ratio=float(k) / np.product(img.shape),
                             alpha=alpha, wvt_level=wvt_level, basis=basis)

    if basis == "dct":
        r_img[:, 0] = r_img[:, 1]
        r_img[0, :] = r_img[1, :]

    r_img[r_img < 0] = 0
    r_img[r_img > 255] = 255

    rmse = utils.compute_rmse(img, r_img)
    print ("rmse: %.3f" % (rmse))
    print ("Sensing rate: %.3f" % (float(k) / np.product(img.shape)))

    if save_path:
        if save_path.endswith(".png"):
            imwrite(save_path, r_img)
            imwrite(save_path[:-4] + "_true.png", img)
        else:
            if alpha is None:
                alpha_s = "best"
            else:
                alpha_s = str(int(alpha * 1e9))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            imwrite(save_path + "/rec_k%d_a%s_r%d.png" %
                   (k, alpha_s, rmse*10000), r_img)
            imwrite(save_path + "/img_k%d_a%s_r%d.png" %
                   (k, alpha_s, rmse*10000), img)
    else:
        plt.clf()
        fig, axarr = plt.subplots(2, 1)

        axarr[0].imshow(img, cmap="gray")
        axarr[1].imshow(r_img, cmap="gray")

        plt.show()


def _sense_thread(args):
    sense_main(img_path=args[0], zoom_rate=args[1], k=args[2], alpha=args[3],
               save_path=args[4], wvt_level=args[5], add_noise=args[6],
               basis=args[7])


def sense_multiple(img_path, ks, alphas, folder, zoom_rate=1., wvt_level=4,
                   add_noise=False, basis="wvt", n_processes=None):
    """ Senses multiple images with k(s) measuerements and alpha(s)

    :param img_path: str
        path to image
    :param ks: list of int
        numbers of measurements
    :param alphas: list of float
        regularization parameter
    :param folder: str
        path to folder to which images will be exported to
    :param zoom_rate: float
        image rescaling factor
    :param wvt_level: int
        level of wavelet transform
    :param add_noise: bool
        adds random noise to the image to reduce SNR
    :param basis: str
        either 'wvt', 'dct'
    :param n_processes:int or None
        number of processes to be used
        if None: all cores are used
    """
    assert os.path.exists(img_path)

    if not n_processes:
        n_processes = cpu_count()

    if not os.path.exists(folder):
        os.makedirs(folder)

    params = []
    for k in ks:
        for a in alphas:
            params.append([img_path, zoom_rate, k, a, folder, wvt_level,
                           add_noise, basis])

    print ("N jobs: %d" % (len(params)))
    if n_processes > 1:
        pool = Pool(n_processes)
        pool.map(_sense_thread, params)
        pool.close()
        pool.join()
    else:
        map(_sense_thread, params)


if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print ("Usage for parameter swiping: python2 sense.py <path_to_img> " \
              "<path_to_save_folder> [<basis>]")
    else:
        img_path = sys.argv[1]
        folder = sys.argv[2]

        if len(sys.argv) == 4:
            basis = sys.argv[3]
            assert basis in ["wvt", "dct"]
        else:
            basis = "wvt"

        ks = [int(ratio*288.**2) for ratio in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25,
                                               0.3, 0.35, 0.4]]
        alphas = np.logspace(-7, 2, num=9)
        
        sense_multiple(img_path, ks, alphas, folder=folder,
                       basis=basis)
       
        utils.compute_rmse_folder(folder)
