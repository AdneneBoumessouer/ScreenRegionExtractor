#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:10:35 2020

@author: adnene
"""

import numpy as np
import cv2
from skimage import filters
from skimage import feature
from skimage import img_as_ubyte
from skimage import color
from preprocess import lines

from scipy.spatial import ConvexHull
from scipy.ndimage.interpolation import rotate


def image_power(image, power=4):
    return image**power


def warp_image(image, pts_src, target_shape):
    # target coordinates
    h, w = target_shape
    pts_dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # tensformation matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # perspective transform
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

# def warp_image_2(image, pts_src, target_shape):
#     # target coordinates
#     h, w = target_shape
#     pts_dst = np.float32([[0,h], [0,0], [w,0], [w,h]])
#     # tensformation matrix
#     M = cv2.getPerspectiveTransform(pts_src,pts_dst)
#     # perspective transform
#     warped = cv2.warpPerspective(image, M, (w,h))
#     return warped


def warp_image_2(image, pts_src, target_shape):
    # target coordinates
    h, w = target_shape
    pts_dst = np.float32([[w, h], [0, h], [0, 0], [w, 0]])
    # tensformation matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # perspective transform
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


# def warp_roi(roi):
#     h, w = roi.shape
#     edged_roi = feature.canny(roi, sigma=3)
#     # boundaries
#     # hlines_roi = lines.detect_hlines(edged_roi)
#     # hline_roi_upper, hline_roi_lower = lines.select_boundaries_from_hlines(hlines_roi, edged_roi.shape)
#     hline_roi_upper = [(0, 0), (w, 0)]
#     hline_roi_lower = [(0, h), (w, h)]
#     vlines_roi = lines.detect_vlines(edged_roi)
#     vline_roi_left, vline_roi_right = lines.select_boundaries_from_vlines(vlines_roi, edged_roi.shape)
#     boundaries_roi = [hline_roi_upper, hline_roi_lower, vline_roi_left, vline_roi_right]
#     # vertices
#     vertices_roi = lines.compute_vertices_coorrdinates(boundaries_roi)
#     # warp roi
#     warped_roi = warp_image(roi, pts_src=vertices_roi, target_shape=roi.shape)
#     warped_roi[warped_roi > 0] = 255
#     return warped_roi

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    Consistent with warp_image_2

    :param points: an nx2 matrix of coordinates
    :rect: an nx2 matrix of coordinates
    """

    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

    # # swap axes ------------------------------
    # rect_tmp = np.zeros_like(rval)
    # rect_tmp[:,0] = rval[:,1]
    # rect_tmp[:,1] = rval[:,0]

    # # compute diagonals
    # diag1 = [rect_tmp[0], rect_tmp[2]]
    # diag2 = [rect_tmp[1], rect_tmp[3]]
    # center = lines.seg_intersect(diag1, diag2)

    # # compute angles
    # angles = []
    # for i in range(4):
    #     diff = rect_tmp[i] - center
    #     re = diff[0]
    #     im = diff[1]
    #     angle = np.angle(re-1j*im, deg=True)
    #     if angle < 0:
    #         angle = angle + 360
    #     angles.append(angles)

    # # sort by angle in ascending order
    # index_list = sorted(range(len(angles)), key=lambda k: angles[k])

    # rect_tmp_2 = np.zeros_like(rect_tmp)
    # rect_tmp_2[0] = rect_tmp[index_list[0]]
    # rect_tmp_2[1] = rect_tmp[index_list[1]]
    # rect_tmp_2[2] = rect_tmp[index_list[2]]
    # rect_tmp_2[3] = rect_tmp[index_list[3]]

    # rect = np.zeros_like(rect_tmp_2)
    # rect[:,0] = rect_tmp_2[:,1]
    # rect[:,1] = rect_tmp_2[:,0]

    # return rect
