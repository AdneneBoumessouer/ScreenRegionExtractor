#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:15:22 2020

Geometry functions

@author: adnene
"""
import numpy as np
from skimage import transform


def detect_hlines(edged):
    # 1. Compute horizontal lines with Hough's method
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    tested_angles = np.linspace(np.pi/2-np.pi/6, np.pi/2+np.pi/6, 120)
    h, theta, d = transform.hough_line(edged, theta=tested_angles)
    h_max = np.amax(h)
    # lines is a list of sublists: a sublist represents a single line and is a list of two tuples representing two points respectively
    hlines = []
    x0 , x1 = 0, edged.shape[1] # x values of the two points
    origin = np.array([x0, x1])
    for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d, min_distance=3, min_angle=2, threshold=0.3*h_max)): # 0.25
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle) # y values of the two points
        pt1 = (x0, y0) # first point
        pt2 = (x1, y1) # second point
        hline = [pt1, pt2] # line as a list of two points
        hlines.append(hline) # lines as a list of lists
    return hlines


def detect_vlines(edged):
    # 1. Compute vertical lines with Hough's method
    tested_angles = np.linspace(-np.pi/6, np.pi/6, 120)
    h, theta, d = transform.hough_line(edged, theta=tested_angles)
    h_max = np.amax(h)
    # lines is a list of sublists: a sublist represents a single line and is a list of two tuples representing two points respectively
    vlines = []
    x0 , x1 = 0, edged.shape[1] # x values of the two points
    origin = np.array([x0, x1])
    for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d, min_distance=3, min_angle=3, threshold=0.3*h_max)): # 0.25
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle) # y values of the two points
        pt1 = (x0, y0) # first point
        pt2 = (x1, y1) # second point
        vline = [pt1, pt2] # line as a list of two points
        vlines.append(vline) # lines as a list of lists
    return vlines


def select_boundaries_from_hlines(hlines, shape):
    # compute all intersections' x-values
    y_axis = [(0.0, 0.0), (0.0, shape[0])]
    intersections_y = []
    for hline in hlines:
        intersections_y.append(seg_intersect(hline, y_axis)[1])
    
    hline_upper_i = np.argmin(np.array(intersections_y))
    hline_lower_i = np.argmax(np.array(intersections_y))
    assert hline_upper_i != hline_lower_i
    
    hline_upper = hlines[hline_upper_i]
    hline_lower = hlines[hline_lower_i]
    return hline_upper, hline_lower


def select_boundaries_from_vlines(vlines, shape):
    # compute all intersections' x-values
    x_axis = [(0.0, 0.0), (shape[1], 0.0)]
    intersections_x = []    
    for vline in vlines:
        intersections_x.append(seg_intersect(vline, x_axis)[0])    
    vline_left_i = np.argmin(np.array(intersections_x))
    vline_right_i = np.argmax(np.array(intersections_x))
    assert vline_left_i != vline_right_i
    
    vline_left = vlines[vline_left_i]
    vline_right = vlines[vline_right_i]
    return vline_left, vline_right

def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(line1, line2):
    # compute intersection coordinates of two line
    # get points
    a1 = np.array(line1[0])
    a2 = np.array(line1[1])
    b1 = np.array(line2[0])
    b2 = np.array(line2[1])
    # compute intersection
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def compute_vertices_coorrdinates(boundaries):
    """
    Returns the 4 vertices by computing intersection of boundaries (lines).

    Parameters
    ----------
    boundaries : list
        List of lists, where each sublist contains two tuples representing two
        points defining a line.

    Returns
    -------
    vertices : numpy array of shape (4, 2)
        coordinates of the vertices, i.e intersection of the four boundaries.
        Order of vertices consistent with transformations.warp_image()

    """
    hline_upper = boundaries[0]
    hline_lower = boundaries[1]
    vline_left = boundaries[2]
    vline_right = boundaries[3]

    vertex_upper_left = seg_intersect(hline_upper, vline_left)
    vertex_upper_right = seg_intersect(hline_upper, vline_right)
    vertex_lower_left = seg_intersect(hline_lower, vline_left)
    vertex_lower_right = seg_intersect(hline_lower, vline_right)
    
    vertices = np.float32([vertex_upper_left, vertex_upper_right, vertex_lower_left, vertex_lower_right])
    return vertices