import numpy as np
import os
from scipy.fftpack import fft
from scipy.misc import imread, imsave
import sklearn.linear_model as lm

import utils
import plotting


def simple_signal(save_path):
    """ Superposition of 3 sinus waves in Time and Frequency domain

    :param save_path: str
        path to folder to which images will be exported
        if None: images will be shown
    """
    n_samples = 1000000
    T = 1.0 / 10000.0

    x = np.linspace(0.0, n_samples * T, n_samples)

    # Create signal - superposition of sinus waves
    y = .5 * np.sin(50.0 * 2.0 * np.pi * x) + \
        .3 * np.sin(200.0 * 2.0 * np.pi * x) + \
        .2 * np.sin(500.0 * 2.0 * np.pi * x)

    xf = np.linspace(0.0, 1.0/(2.0*T), n_samples/2)
    yf = fft(y)
    yf = 2.0 / n_samples * np.abs(yf[0: n_samples / 2])

    if save_path:
        this_save_path = save_path + "/1d_signal.png"
    else:
        this_save_path = None
    plotting.plot_1d(x[:1000], y[:1000], "Time", "Intensity",
                     save_path=this_save_path)

    if save_path:
        if save_path.endswith(".png") or save_path.endswith(".jpg"):
            this_save_path = save_path
        else:
            this_save_path = save_path + "/1d_signal_fft.png"
    else:
        this_save_path = None

    plotting.plot_1d(xf[:n_samples/10], yf[:n_samples/10], "Frequency",
                     "Amplitude", save_path=this_save_path)


def img_signal(img_path, compression_rate=.9, save_path=None, wvt_level=4):
    """ Image and its wavelet transform and naive compression

    :param img_path: str
        path to image
    :param compression_rate: float
        compression rate (0 < c < 1)
    :param save_path: str
        path to folder to which images will be exported
    :param wvt_level: int
        level of wavelet transform
    """
    assert 0 < compression_rate < 1

    img = imread(img_path)

    if len(img.shape) > 2:
        img = np.mean(img, axis=-1)

    off = [(img.shape[0] % 2 ** wvt_level) / 2.,
           (img.shape[1] % 2 ** wvt_level) / 2.]

    img = img[np.floor(off[0]): img.shape[0] - np.ceil(off[0]),
              np.floor(off[1]): img.shape[1] - np.ceil(off[1])]

    if save_path:
        imsave(save_path + os.path.basename(img_path)[:-4] + "_cropped.png",
               img)

    print("Cropped image to: ", img.shape)

    img_dw = utils.dwt2(img, level=wvt_level)

    if save_path:
        this_save_path = save_path + "/img_dwvt.png"
    else:
        this_save_path = None

    plotting.plot_img(img_dw, save_path=this_save_path)

    coeff = img_dw.flatten()
    abs_sorting = np.argsort(np.abs(coeff))

    if save_path:
        this_save_path = save_path + "/abs_coeffs.png"
    else:
        this_save_path = None

    plotting.plot_1d(range(len(coeff))[:2000],
                     np.abs(coeff)[abs_sorting][::-1][:2000],
                     "Coefficients", "Absolute Amplitude",
                     save_path=this_save_path)

    img_dw[np.abs(img_dw) < np.abs(coeff)[abs_sorting]
                    [int(compression_rate * (np.product(img.shape)-1))]] = 0.

    img_idw = utils.idwt2(img_dw, level=wvt_level)

    if save_path:
        this_save_path = save_path + "/img_idwvt.png"
    else:
        this_save_path = None

    if this_save_path:
        imsave(this_save_path, img_idw)
    else:
        plotting.plot_img(img_idw, save_path=this_save_path)

    if save_path:
        this_save_path = save_path + "/img_diff_idwt.png"
    else:
        this_save_path = None

    plotting.plot_img(np.abs(img-img_idw) / img, save_path=this_save_path)


def explain_regularization(save_path=None):
    """ Illustrate effect of Lasso, Ridge, pseudoinverse on underconstrained
    optimization problem

    :param save_path: str
        path to folder to which images will be exported
    """
    # A x = b

    A = np.random.normal(0, 1, 200 * 1000).reshape(200, 1000)
    b = np.random.normal(0, 1, 200)

    clfs = dict(pinv=None,
                lasso=lm.Lasso(alpha=.01),
                ridge=lm.Ridge(alpha=.1))
    xs = {}

    for key in clfs.keys():

        if key == "pinv":
            xs[key] = np.dot(np.linalg.pinv(A), b)
        else:
            clf = clfs[key]
            clf.fit(A, b)
            xs[key] = clf.coef_

        if save_path:
            this_save_path = save_path + "/" + key + ".png"
        else:
            this_save_path = None

        plotting.plot_hist(xs[key], [-.2, -.1, 0., .1, .2],
                           save_path=this_save_path,
                           xlabel="coefficient magnitude",
                           ylabel="density")
        print key, " - Error: %.5f" % np.sum(np.abs(np.dot(A, xs[key]) - b)), \
            "Number of nonzero elements: %d" % len(np.nonzero(xs[key])[0])

