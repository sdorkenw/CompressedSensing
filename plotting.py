from matplotlib import pyplot as plt
import numpy as np


def plot_hist(data, x_ticks=None, xlabel=None, ylabel=None, save_path=None):
    """ Plots a histogram

    :param data: 1d arr
        data
    :param x_ticks: list of str
        x ticks
    :param xlabel: str
        x label
    :param ylabel: str
        y label
    :param save_path: str
        save path
        if None: plot is shown
    """
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')

    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22, direction='out',
                   length=8, width=3., right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='minor', labelsize=22, direction='out',
                   length=0, width=0, right="off", top="off", pad=10)
    ax.tick_params(axis='x', which='minor', labelsize=22, direction='out',
                   length=0, width=0, right="off", top="off", pad=10)

    weights = np.ones_like(data) / len(data)
    plt.hist(data, 50, normed=False, weights=weights, facecolor='k', lw=0)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=26)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=26)

    if x_ticks:
        plt.xticks(x_ticks, x_ticks)

    plt.tight_layout()

    if save_path is None:
        plt.show(block=True)
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_1d(xs, ys, xlabel, ylabel, xlim=None, ylim=None, labels=None,
            save_path=None):
    """ 1D plot

    :param x: list of values
        x values
    :param y: list of values
        y values
    :param xlabel: str
        x label
    :param ylabel: str
        y label
    :param save_path: str
        save path
        if None: plot is shown
    """
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22, direction='out',
                   length=8, width=3., right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='minor', labelsize=22, direction='out',
                   length=0, width=0, right="off", top="off", pad=10)
    ax.tick_params(axis='x', which='minor', labelsize=22, direction='out',
                   length=0, width=0, right="off", top="off", pad=10)

    if not isinstance(xs[0], list):
        xs = [xs]
    if not isinstance(ys[0], list):
        ys = [ys]

    colors = ["k", "0.6", "0.3", "0.8"]

    for i_line in range(len(xs)):
        x = xs[i_line]
        y = ys[i_line]
        if labels is not None:
            ax.plot(x, y, lw=3, c=colors[i_line % len(colors)],
                    label=labels[i_line])
        else:
            ax.plot(x, y, lw=3, c=colors[i_line % len(colors)])

    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)

    if labels is not None:
        legend = ax.legend(loc='best', shadow=False, frameon=False,
                           fontsize=22)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_img(img, save_path=None, vmin=None, vmax=None, cmap="jet"):
    """ 2D plot

    :param img: 2d array
        image
    :param save_path: str
        save path
        if None: plot is shown
    :param vmin: int
        minimum of colormap
    :param vmax: int
        maximum of colormap
    :param cmap: str
        colormap (choose from plt.colormaps())
    """
    if not vmin:
        if np.min(img) < -1:
            vmin = -255
        elif np.min(img) < 0:
            vmin = -1
        else:
            vmin = 0

    if not vmax:
        if np.max(img) > 1:
            vmax = 255
        else:
            vmax = 1

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')

    ax.tick_params(axis='both', which='major', labelsize=22, direction='out',
                   length=0, width=0., right="off", top="off", pad=10)
    ax.tick_params(axis='both', which='minor', labelsize=22, direction='out',
                   length=0, width=0, right="off", top="off", pad=10)

    plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    cm = plt.colorbar()
    cm.ax.tick_params(labelsize=20)
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()
