########################################################################
#
# File:   project2.py
# Author: Alan Zhuolun Zhao, Pepper Shiqin Wang
# Instructor: Matt Zucker
# Date:   February, 2017
#
# Written for ENGR 27 - Computer Vision
#
########################################################################

from __future__ import print_function

import cv2
import numpy
import sys

def fixKeyCode(code):
    return numpy.uint8(code).view(numpy.int8)

if len(sys.argv) != 2:
    print('usage: {} IMAGE'.format(sys.argv[0]))
    sys.exit(1)

filename = sys.argv[1]
original = cv2.imread(filename)

def pyr_build(img):
    G = []
    G_plus = []
    L = []
    G.append(img)
    w = img.shape[1]
    h = img.shape[0]
    min_dimension = min(w, h)
    i = 0
    while min_dimension > 16:
        new_img = cv2.pyrdown(G[i])
        G.append(new_img)
        new_img_up = cv2.pyrup(new_img, Size(w, h))
        G_plus.append(new_img_up)
        G_32 = numpy.array(G[i], dtype = 'float32')
        new_img_up_32 = numpy.array(G[i], dtype = 'float32')
        result = cv2.absdiff(G_32, new_img_up_32)
        L.append(result)
        w = new_img.shape[1]
        h = new_img.shape[0]
        min_dimension = min(w, h)
        i += 1

    return L


cv2.namedWindow('original')
cv2.imshow('original', original)

while fixKeyCode(cv2.waitKey(15)) < 0:
    pass
