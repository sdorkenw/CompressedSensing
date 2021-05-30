import glob
import numpy as np
import os
import pywt
from imageio import imread
from scipy.fftpack import dct, idct


def decrease_SNR(img, rate=0.05):
    """ Reduces SNR

    :param img: 2d array
        image
    :param rate: float
        rate of noise
    :return: 2d array
        noisy image
    """
    img_n = img.copy().astype(np.float32)
    if np.max(img_n) > 1:
        img_n /= 255
    return img_n + rate * np.random.randn(*img.shape)


def compute_rmse(true_img, rec_img):
    """ Computes rmse for an image regarding its true image

    :param true_img: 2d array of floats or ints
        scale: 0-1 or 0-255
    :param rec_img: 2d array of floats or ints
        scale: 0-1 or 0-255
    :return: float
        rmse
    """
    if np.max(true_img) > 1:
        true_img = true_img.astype(np.float32) / 255
    if np.max(rec_img) > 1:
        rec_img = rec_img.astype(np.float32) / 255
    diff = true_img - rec_img
    return np.sqrt(np.mean(np.power(diff, 2)))


def compute_rmse_folder(folder, start_true="img", start_rec="rec"):
    """ Computes rmse for many images in a folder and outputs them in a file

    :param folder: str
        folder with images
    :param start_true: str
        start of basename of true images
    :param start_rec: str
        start of basename of reconstructed images belonging to the true images
    """
    paths = glob.glob(folder + "/%s*" % start_true)
    best = [1000, ""]
    with open(folder + "/rmse.txt", "w") as f:
        for path in paths:
            rec_path = os.path.dirname(path) + "/%s" % start_rec + \
                       os.path.basename(path)[len(start_true):]
            rmse = compute_rmse(imread(path),
                                imread(rec_path))
            f.write("%s: %.6f\n" % (os.path.basename(path)[4:], rmse))

            if rmse < best[0]:
                best[0] = rmse
                best[1] = os.path.basename(path)[4:]

        f.write("\n\nBest:\n--------\n")
        f.write("%s: %.6f\n" % (best[1], best[0]))


def _from_coeffs_to_img(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))


def _split_coeffs(Wimg):
    L1, L2 = np.hsplit(Wimg, 2)
    LL, HL = np.vsplit(L1, 2)
    LH, HH = np.vsplit(L2, 2)
    return LL, [LH, HL, HH]


def _from_img_to_coeffs(Wimg, levels=1):
    LL, coeff = _split_coeffs(Wimg)
    coeffs = [coeff]
    for i in range(levels - 1):
        LL, c = _split_coeffs(LL)
        coeffs.insert(0, c)
    coeffs.insert(0, LL)
    return coeffs


def dwt2(img, level=4):
    """ 2d wavelet transformation

    :param img: 2d array
        input image
    :param level: int
        level of wavelet transform - image shape has to be multiples of 2**level
    :return: 2d array
        wavelet coefficients
    """
    coeffs = pywt.wavedec2(img, wavelet='db4', mode='per', level=level)
    Wimg, r = coeffs[0], coeffs[1:]
    for levels in r:
        Wimg = _from_coeffs_to_img(Wimg, levels)
    return Wimg


def idwt2(Wimg, level=4):
    """ inverse 2d wavelet transform

    :param Wimg: 2d array
        wavelet coefficients
    :param level: int
        level of wavelet transform - image shape has to be multiples of 2**level
    :return: 2d array
        image
    """
    coeffs = _from_img_to_coeffs(Wimg, levels=level)
    return pywt.waverec2(coeffs, wavelet='db4', mode='per')


def dct2(img, t=3):
    """ 2d discrete cosine transform

    :param img: 2d array
        image
    :param t: int
        type of dct
    :return: 2d array
        dct coefficients
    """
    return dct(dct(img.T, type=t).T, type=t)


def idct2(Dimg, t=3):
    """ inverse 2d discrete cosine transform

    :param Dimg: 2d array
        dct coefficients
    :param t: int
        type of dct
    :return: 2d array
        image
    """
    return idct(idct(Dimg.T, type=t).T, type=t)


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