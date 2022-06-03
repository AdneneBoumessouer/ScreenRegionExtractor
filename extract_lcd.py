#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:41:28 2022

@author: aboumessouer
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, util, filters, transform, draw, color
from preprocess import utils, transformation, features

# DEFAULT_IMAGE_SHAPE = (450, 800, 3)


def extract_display(img,  img_name=None, shape=(450, 800, 3), plot=False, save_dir=None):
    # reshape image
    if shape is not None:
        img = transform.resize(img, output_shape=shape)

    # RGB to HSV
    img_hsv = color.rgb2hsv(img)

    #
    if plot:
        fig, axarr = plt.subplots(
            nrows=4, ncols=5, figsize=(35, 20))  # (35,15)
        axarr[0, 0].imshow(img, cmap=None)
        axarr[0, 0].set_title("original image")
        axarr[0, 1].set_axis_off()
        axarr[0, 2].set_axis_off()
        axarr[0, 3].set_axis_off()
        axarr[0, 4].imshow((255*np.ones(shape=(450, 200, 3))).astype("uint8"))
        axarr[0, 4].set_axis_off()

    channel_names = ["HUE", "SATURATION", "VALUE"]
    extents, contours, masks, masks_rect = [], [], [], []

    for i in range(3):
        # threshold hsv-channel
        img_ch = img_hsv[:, :, i]
        img_ch = filters.gaussian(img_ch, sigma=2)
        th = filters.threshold_otsu(filters.gaussian(img_ch, sigma=2))
        binary = img_ch > th
        binary = utils.pad_binary_image(binary, dh=20, dw=20)

        # find contours and select largest contour as probable candidate for LCD display
        cnts = measure.find_contours(util.img_as_float(binary), level=0.9)
        m = np.argmax([cnt.shape[0] for cnt in cnts])
        cnt_max = cnts[m]
        contours.append(cnt_max)

        # compute convex hull binary masks
        mask = draw.polygon2mask(binary.shape[0:2], cnt_max)
        masks.append(mask)

        # compute minimum rotated bounding box
        rect = transformation.minimum_bounding_rectangle(cnt_max)
        mask_rect = draw.polygon2mask(binary.shape[0:2], rect)
        masks_rect.append(mask_rect)

        # compute extent: ratio of pixels in the region to pixels in the total rotated bounding box
        extent = features.compute_extent(binary, cnt_max, rect)
        extents.append(extent)

        # plot ---------------------------------------------------------------
        # hsv channel
        if plot:
            channel = axarr[i+1, 0].imshow(img_ch, cmap="inferno")
            fig.colorbar(channel, ax=axarr[i+1, 0])
            axarr[i+1, 0].set_title(channel_names[i])

            # binary + low tolerance
            axarr[i+1, 1].imshow(binary, cmap="gray")
            axarr[i+1, 1].plot(cnt_max[:, 1],
                               cnt_max[:, 0], color="r")
            axarr[i+1, 1].set_title("binary + contour")

            # contour + mask_rect + rect_vertices
            axarr[i+1, 2].imshow(0.7*util.img_as_float(mask_rect),
                                 vmin=0.0, vmax=1.0, cmap="gray")
            axarr[i+1, 2].plot(cnt_max[:, 1],
                               cnt_max[:, 0], color="r")
            axarr[i+1, 2].scatter(rect[:, 1], rect[:, 0], linewidths=4)
            axarr[i+1, 2].set_title("contour + mask_rect + rect_vertices")

    # get the channel which contains LCD display
    j = np.argmax(extents)
    # retireve contour and mask which yielded best score (i.e extent)
    contour, mask, mask_rect = contours[j], masks[j], masks_rect[j]

    if plot:
        for i in range(3):
            if i == j:
                # green flag
                flag = np.zeros(shape=(450, 200, 3), dtype=np.uint8)
                flag[:, :] = [154, 205, 50]
            else:
                # red flag
                flag = np.zeros(shape=(450, 200, 3), dtype=np.uint8)
                flag[:, :] = [255, 0, 0]

            axarr[i+1, 3].imshow(flag)
            axarr[i+1, 3].set_axis_off()
            axarr[i+1, 3].set_title("extent = {:.3f}".format(extents[i]))

    # extract lcd
    img_lcd = utils.pad_image(img)
    # img_lcd[np.dstack(3*[~mask])] = 0.0
    img_lcd[np.dstack(3*[~mask_rect])] = 0.0

    # plot -------------------------------------------------------------------
    if plot:
        axarr[j+1, 4].imshow(img_lcd)
        axarr[j+1, 4].set_title("img_lcd")
        plt.tight_layout()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "rect_"+str(img_name)+".png"))
        plt.close("all")
    # print(k, img_name)

    return img_lcd, mask_rect.astype(np.uint8), contour
