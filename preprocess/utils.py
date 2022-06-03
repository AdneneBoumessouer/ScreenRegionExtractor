#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:50:21 2020

Utilitary Functions

@author: adnene
"""

import numpy as np


def has_white_background(binary, ph=0.1, pw=0.05):
    H, W = binary.shape
    dH = int(round(ph*H))
    dW = int(round(pw*W))
    mask = np.ones_like(binary, dtype="bool")
    mask[dH:H-dH, dW:W-dW] = False
    background = binary[mask]
    if np.count_nonzero(background)/len(background) >= 0.5:
        return True
    return False


def pad_binary_image(binary, dh=20, dw=20):
    assert dh > 0
    assert dw > 0
    h, w = binary.shape
    if has_white_background(binary):
        binary_stretched = np.ones(shape=(h + 2*dh, w + 2*dw), dtype="bool")
    else:
        binary_stretched = np.zeros(shape=(h + 2*dh, w + 2*dw), dtype="bool")
    binary_stretched[dh:h+dh, dw:w+dw] = binary
    return binary_stretched


def pad_image(image, dh=20, dw=20):
    assert dh > 0
    assert dw > 0
    h, w, c = image.shape
    image_stretched = np.zeros_like(image, shape=(h + 2*dh, w + 2*dw, c))
    image_stretched[dh:h+dh, dw:w+dw, :] = image
    return image_stretched
