#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:51:15 2020

@author: adnene
"""

import numpy as np

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
    }


def get_seven_segments_from_roi(roi_shape, bounding_rect):
    # compute the width and height of each of the 7 segments
    # we are going to examine
    (roiH, roiW) = roi_shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)
    
    # extract the digit ROI
    (x, y, w, h) = bounding_rect
    
    # define the set of 7 segments
    segments = [
        ((0, 0), (w, dH)),	# top
        ((0, 0), (dW, h // 2)),	# top-left
        ((w - dW, 0), (w, h // 2)),	# top-right
        ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
        ((0, h // 2), (dW, h)),	# bottom-left
        ((w - dW, h // 2), (w, h)),	# bottom-right
        ((0, h - dH), (w, h))	# bottom
        ]
    return segments

def generate_expected_digit_roi(roi_shape, segments):
    # define the dictionary of digit segments so we can identify
    # each digit on the thermostat    
    rois_digits = []    
    for code in list(DIGITS_LOOKUP.keys()):
        roi_d = np.zeros(shape=roi_shape, dtype="uint8")
        on_i = np.nonzero(code)[0]
        segments_on = np.array(segments)[on_i]
        for segment_on in segments_on:
            x1 = segment_on[0][0]
            x2 = segment_on[1][0]
            y1 = segment_on[0][1]
            y2 = segment_on[1][1]
            roi_d[y1:y2, x1:x2] = 255
        rois_digits.append(roi_d)
    return rois_digits