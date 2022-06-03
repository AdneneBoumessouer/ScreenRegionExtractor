#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:57:20 2020

@author: adnene
"""

import numpy as np
from skimage import draw


def compute_extent(binary, coords_low_precision, coords_high_precision):
    # create masks for region inside low and high contours
    mask_low = draw.polygon2mask(binary.shape, coords_low_precision)
    mask_high = draw.polygon2mask(binary.shape, coords_high_precision)
    # intersection of low and high masks
    mask_inter = np.zeros_like(binary, shape=binary.shape)
    mask_union = np.zeros_like(binary, shape=binary.shape)
    # union of low and high masks
    mask_inter[(mask_low == True) & (mask_high == True)] = True        
    mask_union[(mask_low == True) | (mask_high == True)] = True
    
    # compute IoU: Intersection over Union
    if np.count_nonzero(mask_union) == 0:
        extent = 0.0
    else:
        extent = np.count_nonzero(mask_inter) / np.count_nonzero(mask_union)
    return extent