from matplotlib import pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
from scipy.misc import imread, imsave
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
    img_f = img.flatten()
    print "Build sensing matrix"
    A = np.random.normal(0, 1, len(img_f)*k).astype(np.float32).\
        reshape(k, img.shape[0], img.shape[1])

    print "Measurement"
    b = np.dot(A.reshape(k, len(img_f)), img_f)

    if basis == "wvt":
        print "Wavelets"
        trans_A = [utils.dwt2(A[i].reshape(img.shape), level=wvt_level).
                     astype(np.float16).flatten() for i in range(k)]
    elif basis == "dct":
        print "DCT"
        trans_A = [utils.dct2(A[i].reshape(img.shape)).
                       astype(np.float16).flatten() for i in range(k)]
    else:
        raise Exception("Unknown basis")

    A = None

    if alpha:
        lasso = lm.LassoLars(alpha=alpha, max_iter=100000, normalize=True)
        print "Fit"
        lasso.fit(trans_A, b)
    else:
        lasso_cv = lm.LassoLarsCV(n_jobs=cpu_count(), max_iter=100000,
                              normalize=True)
        print "Fit"
        lasso_cv.fit(trans_A, b)
        print "Alpha: %.6f" % lasso_cv.alpha_
        lasso = lm.LassoLars(alpha=lasso_cv.alpha_, max_iter=100000,
                         normalize=True)
        print "Fit"
        lasso.fit(trans_A, b)

    if basis == "wvt":
        return utils.idwt2(lasso.coef_.reshape(img.shape), level=wvt_level)
    elif basis == "dct":
        return utils.idct2(lasso.coef_.reshape(img.shape))
    else:
        raise Exception("Unknown basis")


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
        print "Only grayscale supported..."
        img = np.mean(img, axis=-1)
        print "... RGB converted to grayscale"

    if zoom_rate != 1:
        img = scipy.ndimage.zoom(img, zoom=zoom_rate, order=3)

    if basis == "wvt":
        off = [(img.shape[0] % 2 ** wvt_level) / 2.,
               (img.shape[1] % 2 ** wvt_level) / 2.]

        img = img[int(np.floor(off[0])): img.shape[0] - int(np.ceil(off[0])),
                  int(np.floor(off[1])): img.shape[1] - int(np.ceil(off[1]))]

        print "Cropped image to: ", img.shape

        img = img.astype(np.float16) / np.max(img)

    if add_noise:
        r_img = sense(utils.decrease_SNR(img), k=k, alpha=alpha,
                      wvt_level=wvt_level, basis=basis)
    else:
        r_img = sense(img.copy(), k=k, alpha=alpha, wvt_level=wvt_level,
                      basis=basis)
    if basis == "dct":
        r_img[:, 0] = r_img[:, 1]
        r_img[0, :] = r_img[1, :]
    r_img[r_img < 0] = 0

    rmse = utils.compute_rmse(img, r_img)
    print "rmse: %.3f" % (rmse)
    print "Sensing rate: %.3f" % (float(k) / np.product(img.shape))

    if save_path:
        if save_path.endswith(".png"):
            imsave(save_path, r_img)
            imsave(save_path[:-4] + "_true.png", img)
        else:
            alpha_s = int(alpha * 1e9)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            imsave(save_path + "/rec_k%d_a%d_r%d.png" %
                   (k, alpha_s, rmse*10000), r_img)
            imsave(save_path + "/img_k%d_a%d_r%d.png" %
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

    print "N jobs: %d" % (len(params))
    if n_processes > 1:
        pool = Pool(n_processes)
        pool.map(_sense_thread, params)
        pool.close()
        pool.join()
    else:
        map(_sense_thread, params)


if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print "Usage for parameter swiping: python2 sense.py <path_to_img> " \
              "<path_to_save_folder> [<basis>]"
    else:
        img_path = sys.argv[1]
        folder = sys.argv[2]

        if len(sys.argv) == 4:
            basis = sys.argv[3]
            assert basis in ["wvt", "dct"]
        else:
            basis = "wvt"

        # ks = [8000, 16000, 24000]
        ks = [int(ratio*288.**2) for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]]
        alphas = np.logspace(-7, 0, num=8)
        sense_multiple(img_path, ks, alphas, folder=folder,
                       basis=basis)
        utils.compute_rmse_folder(folder)
