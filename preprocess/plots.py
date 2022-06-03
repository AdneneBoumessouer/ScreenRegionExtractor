#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:05:55 2020

@author: adnene
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color, exposure, filters
from preprocess import utils


def plot_hsv_hist_binary(img, figsize=(21, 15)):
    """
    Takes RGB image, converts to HSV-colorspace and plots HSV channels along with
    pixel intensity histograms and thresholded images (with otsu method).

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    figsize : TYPE, optional
        DESCRIPTION. The default is (21,15).

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    # convert to hsv
    img_hsv = color.rgb2hsv(img)

    fig, axarr = plt.subplots(nrows=4, ncols=3, figsize=figsize)
    axarr[0, 0].imshow(img, cmap=None)
    axarr[0, 0].set_title("ORIG")
    axarr[0, 2].imshow(img_hsv, cmap=None)
    axarr[0, 2].set_title("HSV")

    titles = ["HUE", "SATURATION", "VALUE"]
    for i in range(3):
        # HUE
        img_ch = img_hsv[:, :, i]
        hist, bins = exposure.histogram(
            img_ch, nbins=256, source_range="image")
        # img
        hue = axarr[i+1, 0].imshow(img_ch, vmin=0.0, vmax=1.0, cmap="inferno")
        fig.colorbar(hue, ax=axarr[i+1, 0])
        axarr[i+1, 0].set_title(titles[i])
        del hue
        # hist
        axarr[i+1, 1].plot(bins, hist)
        axarr[i+1, 1].set_title("histogram")
        # thresh
        th = filters.threshold_otsu(img_ch)
        binary = img_ch > th
        binary = utils.pad_binary_image(binary)
        axarr[i+1, 1].vlines(x=th, ymin=0.0, ymax=np.amax(hist), colors="r")
        axarr[i+1, 2].imshow(binary, cmap="gray")
        axarr[i+1, 2].set_title("binary")

    plt.tight_layout()
    return fig
