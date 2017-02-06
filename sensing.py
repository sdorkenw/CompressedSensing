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
        lasso = lm.Lasso(alpha=alpha, max_iter=100000, normalize=True)
        print "Fit"
        lasso.fit(trans_A, b)
    else:
        lasso_cv = lm.LassoCV(n_jobs=cpu_count(), max_iter=100000,
                              normalize=True)
        print "Fit"
        lasso_cv.fit(trans_A, b)
        print "Alpha: %.6f" % lasso_cv.alpha_
        lasso = lm.Lasso(alpha=lasso_cv.alpha_, max_iter=100000,
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
               basis="wvt", save_path=None):
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

    off = [(img.shape[0] % 2 ** wvt_level) / 2.,
           (img.shape[1] % 2 ** wvt_level) / 2.]

    img = img[int(np.floor(off[0])): img.shape[0] - int(np.ceil(off[0])),
              int(np.floor(off[1])): img.shape[1] - int(np.ceil(off[1]))]

    print "Cropped image to: ", img.shape

    img = img.astype(np.float16) / np.max(img)

    r_img = sense(img.copy(), k=k, alpha=alpha, wvt_level=wvt_level,
                  basis=basis)
    r_img[r_img < 0] = 0

    print "rmse: %.3f" % (utils.compute_rmse(img, r_img))
    print "Sensing rate: %.3f" % (float(k) / np.product(img.shape))

    if save_path:
        if save_path.endswith(".png"):
            imsave(save_path, r_img)
            imsave(save_path[:-4] + "_true.png", img)
        else:
            alpha_s = int(alpha * 1e9)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            imsave(save_path + "/rec_k%d_a%d.png" % (k, alpha_s), r_img)
            imsave(save_path + "/img_k%d_a%d.png" % (k, alpha_s), img)
    else:
        plt.clf()
        fig, axarr = plt.subplots(2, 1)

        axarr[0].imshow(img, cmap="gray")
        axarr[1].imshow(r_img, cmap="gray")

        plt.show()


def _sense_thread(args):
    sense_main(img_path=args[0], k=args[1], alpha=args[2], wvt_level=args[3],
               basis=args[4], save_path=args[5])


def sense_multiple(ks, alphas, folder, img_path, wvt_level=4, basis="wvt",
                   n_processes=None):
    """ Senses multiple images with k(s) measuerements and alpha(s)

    :param ks: list of int
        numbers of measurements
    :param alphas: list of float
        regularization parameter
    :param folder: str
        path to folder to which images will be exported to
    :param img_path: str
        path to image
    :param wvt_level: int
        level of wavelet transform
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
            params.append([img_path, k, a, wvt_level, folder])

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

        ks = [8000, 16000, 24000]
        alphas = np.logspace(-7, 0, num=8)
        sense_multiple(ks, alphas, folder=folder, img_path=img_path,
                       basis=basis)
        utils.compute_rmse_folder(folder)
