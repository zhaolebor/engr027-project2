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

########################################################################
def fixKeyCode(code):
    return numpy.uint8(code).view(numpy.int8)

def pyr_build(img):
    G = []
    L = []
    G.append(img)
    w = img.shape[1]
    h = img.shape[0]
    min_dimension = min(w, h)
    i = 0
    while min_dimension > 16:
        new_img = cv2.pyrDown(G[i])
        G.append(new_img)
        new_img_up = cv2.pyrUp(new_img, dstsize = (w, h))
        G_32 = numpy.array(G[i], dtype = 'float32')
        new_img_up_32 = numpy.array(new_img_up, dtype = 'float32')

        result = G_32 - new_img_up_32
        #result = cv2.absdiff(G_32, new_img_up_32)
        L.append(result)
        w = new_img.shape[1]
        h = new_img.shape[0]
        min_dimension = min(w, h)
        i += 1

    return L

def pyr_reconstruct(L):
    n = len(L)-1
    R = []
    R.append(L[n])
    i = n
    while i > 0:
        w = L[i-1].shape[1]
        h = L[i-1].shape[0]
        r_up = cv2.pyrUp(R[n-i], dstsize = (w, h))
        r_up_32 = numpy.array(r_up, dtype = 'float32')
        l_32 = numpy.array(L[i-1], dtype = 'float32')
        new_r = r_up_32 + l_32
        R.append(new_r)
        i -= 1

    return R[n]


########################################################################
if len(sys.argv) != 2:
    print('usage: {} IMAGE'.format(sys.argv[0]))
    sys.exit(1)

filename = sys.argv[1]
original = cv2.imread(filename)

cv2.namedWindow('original')
cv2.imshow('original', original)
print(original)

lp = pyr_build(original)
#counter = 0
#for item in lp:
#    cv2.imshow('pyramid', 0.5 + 0.5*(item / numpy.abs(item).max()))
#    cv2.imwrite('pyramid'+str(counter)+'.jpg', item)
#    counter += 1
#    while fixKeyCode(cv2.waitKey(15)) < 0:
#        pass

convert = pyr_reconstruct(lp)
convert_clip = numpy.clip(convert, 0, 255)
convert_8 = numpy.array(convert_clip, dtype = 'uint8')
print(convert_8)
cv2.imshow('convert', convert_8)

while fixKeyCode(cv2.waitKey(15)) < 0:
    pass
