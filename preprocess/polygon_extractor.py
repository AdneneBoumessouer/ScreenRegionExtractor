#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:20:37 2020

@author: adnene
"""

from skimage import filters

class PolygonExtractor:
    
    def __init__(self):
        pass
    
    def compute_score(self, img_ch):
        pass
    
    def channel2binary(self, img_ch):
        img_ch = img_hsv[:, :, i]
        img_ch = filters.gaussian(img_ch, sigma=2)
        th = filters.threshold_otsu(img_ch)
        binary = img_ch > th
        binary = stretch_binary_image(binary, dh=20, dw=20)
        return binary