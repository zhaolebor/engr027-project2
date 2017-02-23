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
    while min_dimension > 8:
        new_img = cv2.pyrDown(G[i])
        G.append(new_img)
        new_img_up = cv2.pyrUp(new_img, dstsize = (w, h))
        G_32 = numpy.array(G[i], dtype = 'float32')
        new_img_up_32 = numpy.array(new_img_up, dtype = 'float32')

        result = G_32 - new_img_up_32
        L.append(result)
        w = new_img.shape[1]
        h = new_img.shape[0]
        min_dimension = min(w, h)
        i += 1

    return L

def pyr_reconstruct(L):
    n = len(L)-1
    R = []
    new_l = 127 + 127 * (L[n] / numpy.abs(L[n]).max())
    R.append(new_l)
    i = n
    while i > 0:
        w = L[i-1].shape[1]
        h = L[i-1].shape[0]
        r_up = cv2.pyrUp(R[n-i], dstsize = (w, h))
        #We don't need to convert to float32 because they already were
        #r_up_32 = numpy.array(r_up, dtype = 'float32')
        #l_32 = numpy.array(L[i-1], dtype = 'float32')
        new_r = r_up + L[i-1]
        R.append(new_r)
        i -= 1

    return R[n]

def alpha_blend(A, B, alpha):
    #print(A.shape)
    #print(B.shape)
    #print(alpha.shape)
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = numpy.expand_dims(alpha, 2)
    return A + alpha*(B-A)

def LP_alpha_blend(A, B, alpha):
    lp_a = pyr_build(A)
    lp_b = pyr_build(B)
    blend = []
    for i in range(len(lp_a)):
        w = lp_a[i].shape[1]
        h = lp_a[i].shape[0]
        alpha_new = cv2.resize(alpha, dsize = (w, h), interpolation = cv2.INTER_AREA)
        blend_new = alpha_blend(lp_a[i], lp_b[i], alpha_new)
        blend.append(blend_new)
    result = pyr_reconstruct(blend)
    return result

def align(A, B):
    w_a = A.shape[1]
    h_a = A.shape[0]
    w_b = B.shape[1]
    h_b = B.shape[0]
    if w_a != w_b or h_a != h_b:
        new_B = cv2.resize(B, dsize = (w_a, h_a), interpolation = cv2.INTER_AREA)
    else:
        new_B = B
    return new_B


########################################################################
if sys.argv[1] == 'pyramid':
    if len(sys.argv) != 3:
        print('usage: {} pyramid IMAGE1'.format(sys.argv[0]))
        sys.exit(1)

    filename = sys.argv[2]
    original = cv2.imread(filename)
    cv2.namedWindow('original')
    cv2.imshow('original', original)

    lp = pyr_build(original)

    counter = 0
    for item in lp:
        cv2.imshow('pyramid', 0.5 + 0.5*(item / numpy.abs(item).max()))
        cv2.imwrite('pyramid'+str(counter)+'.jpg', item)
        counter += 1
        while fixKeyCode(cv2.waitKey(15)) < 0:
            pass

    convert = pyr_reconstruct(lp)
    convert_clip = numpy.clip(convert, 0, 255)
    convert_8 = numpy.array(convert_clip, dtype = 'uint8')
    print(convert)
    print(convert_8)
    print(original)
    cv2.imshow('convert', convert_8)

elif sys.argv[1] == 'blend':
    if len(sys.argv) != 4:
        print('usage: {} blend IMAGE1 IMAGE2'.format(sys.argv[0]))
        sys.exit(1)

    filename1 = sys.argv[2]
    filename2 = sys.argv[3]
    A = cv2.imread(filename1)
    B = cv2.imread(filename2)

    B = align(A, B)

    width = A.shape[1]
    height = A.shape[0]
    cx = width/2
    cy = height/2
    angle = 0
    sigma = 0

    mask = numpy.zeros((height, width), dtype=numpy.uint8)
    cv2.ellipse(mask, (cx, cy), (width/4, height/4), angle, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    mask_blurred = cv2.GaussianBlur(mask, (5,5), sigma)
    alpha = mask.astype(numpy.float32) / 255.0

    cv2.namedWindow('alpha')
    cv2.imshow('alpha', alpha)

    blend = alpha_blend(A, B, alpha)

    blend_clip = numpy.clip(blend, 0, 255)
    blend_8 = numpy.array(blend_clip, dtype = 'uint8')
    cv2.namedWindow('blend')
    cv2.imshow('blend', blend_8)

    pyramid_blend = LP_alpha_blend(A, B, alpha)

    pyramid_blend_clip = numpy.clip(pyramid_blend, 0, 255)
    pyramid_blend_8 = numpy.array(pyramid_blend_clip, dtype = 'uint8')
    cv2.namedWindow('pyramid blend')
    cv2.imshow('pyramid blend', pyramid_blend_8)


elif sys.argv[1] == 'hybrid':
    if len(sys.argv) != 4:
        print('usage: {} blend IMAGE1 IMAGE2'.format(sys.argv[0]))
        sys.exit(1)

    filename1 = sys.argv[2]
    filename2 = sys.argv[3]
    A = cv2.imread(filename1)
    B = cv2.imread(filename2)

    B = align(A, B)

    k_a = 1
    k_b = 1
    sigma_a = 10
    sigma_b = 5

    A_low = cv2.GaussianBlur(A, (5,5), sigma_a).astype('float32')
    B_low = cv2.GaussianBlur(B, (5,5), sigma_b).astype('float32')
    B_high = B - B_low

    I = k_a * A_low + k_b * B_high

    I_clip = numpy.clip(I, 0, 255)
    I_8 = numpy.array(I_clip, dtype = 'uint8')
    cv2.namedWindow('hybrid')
    cv2.imshow('hybrid', I_8)



else:
    print('usage: {} pyramid IMAGE1'.format(sys.argv[0]))
    print('usage: {} blend IMAGE1 IMAGE2'.format(sys.argv[0]))
    print('usage: {} hybrid IMAGE1 IMAGE2'.format(sys.argv[0]))

    sys.exit(1)





while fixKeyCode(cv2.waitKey(15)) < 0:
    pass
