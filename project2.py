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
import cvk2

########################################################################
def fixKeyCode(code):
    return numpy.uint8(code).view(numpy.int8)

# function to build up a Lapacian pyramid with an image input
def pyr_build(img):
    G = []
    L = []
    G.append(img.astype('float32'))
    w = img.shape[1]
    h = img.shape[0]
    min_dimension = min(w, h)
    i = 0
    while min_dimension > 8:
        new_img = cv2.pyrDown(G[i])
        G.append(new_img)
        new_img_up = cv2.pyrUp(new_img, dstsize = (w, h))

        result = G[i] - new_img_up
        L.append(result)
        w = new_img.shape[1]
        h = new_img.shape[0]
        min_dimension = min(w, h)
        i += 1

    L[-1] = G[-2]
    return L

# function to reconstruct an image using a Laplacian pyramid as an inpu
def pyr_reconstruct(L):
    n = len(L)-1
    R = []
    R.append(L[n])
    i = n
    while i > 0:
        w = L[i-1].shape[1]
        h = L[i-1].shape[0]
        r_up = cv2.pyrUp(R[n-i], dstsize = (w, h))
        new_r = r_up + L[i-1]
        R.append(new_r)
        i -= 1

    return R[n]

# function to alpha blend two images of the same size
def alpha_blend(A, B, alpha):
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = numpy.expand_dims(alpha, 2)
    return A + alpha*(B-A)

# function to build up two pyramids for two images, alpha blend each layer of the pyramids and reconstruct a new image
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

# function to align image B with the size of image A
def align(A, B):
    w_a = A.shape[1]
    h_a = A.shape[0]
    w_b = B.shape[1]
    h_b = B.shape[0]
    #if w_a != w_b or h_a != h_b:
    #    B = cv2.resize(B, dsize = (w_a, h_a), interpolation = cv2.INTER_AREA)

    pointsA = pickPoints('Image A', A, 'a.txt')
    print('got pointsA =\n', pointsA)

    pointsB = pickPoints('Image B', B, 'b.txt', xcoord=A.shape[1])
    print('got pointsB =\n', pointsB)

    M = cv2.getAffineTransform(pointsB,pointsA)
    B = cv2.warpAffine(B,M,(w_a,h_a))

    return A,B

# function to generate a nice horizontal layout of a pyramid
def pretty(images):
    width = sum(image.shape[1] for image in images)
    height = max(image.shape[0] for image in images)
    output = numpy.zeros((height,width,3))

    y = 0
    for image in images:
        image = 0.5 + 0.5*(image / numpy.abs(image).max())
        h,w,d = image.shape
        output[0:h,y:y+w] = image
        y += w

    return output

# pick points function from cvk2 library
def pickPoints(window, image, filename, xcoord=0):

    cv2.namedWindow(window)
    cv2.imshow(window, image)
    cv2.moveWindow(window, xcoord, 0)

    w = cvk2.MultiPointWidget()

    if w.load(filename):
        print('loaded points from {}'.format(filename))
    else:
        print('could not load points from {}'.format(filename))

    ok = w.start(window, image)

    if not ok:
        print('user canceled instead of picking points')
        sys.exit(1)

    w.save(filename)

    return w.points

# Helper function to calculate distance between two points
def distance(p0, p1):
    return numpy.sqrt(numpy.square(p0[0] - p1[0]) + numpy.square(p0[1] - p1[1]))

########################################################################
# pyramid functionality
if sys.argv[1] == 'pyramid':
    if len(sys.argv) != 3:
        print('usage: {} pyramid IMAGE1'.format(sys.argv[0]))
        sys.exit(1)

    filename = sys.argv[2]
    original = cv2.imread(filename)
    cv2.namedWindow('original')
    cv2.imshow('original', original)

    lp = pyr_build(original)

    # display the pyramid
    output = pretty(lp)
    cv2.imshow('pyramid', output)
    # Since imwrite does not take floating point image, convert to [0,255]
    output = output*255
    output = output.astype(numpy.uint8)
    cv2.imwrite('result/pyramid.jpg', output)

    # reconstruct the original image using the pyramid generated above
    convert = pyr_reconstruct(lp)
    convert_clip = numpy.clip(convert, 0, 255)
    convert_8 = numpy.array(convert_clip, dtype = 'uint8')
    cv2.imshow('convert', convert_8)
    cv2.imwrite('result/convert.jpg', convert_8)

# blend functionality
elif sys.argv[1] == 'blend':
    if len(sys.argv) != 4:
        print('usage: {} blend IMAGE1 IMAGE2'.format(sys.argv[0]))
        sys.exit(1)

    filename1 = sys.argv[2]
    filename2 = sys.argv[3]
    A = cv2.imread(filename1)
    B = cv2.imread(filename2)

    A, B = align(A, B)

    cv2.namedWindow('A')
    cv2.imshow('A', A)
    cv2.namedWindow('B')
    cv2.imshow('B', B)
    while fixKeyCode(cv2.waitKey(15)) < 0:
        pass

    width = A.shape[1]
    height = A.shape[0]

    # ask the user if they want to pick points on the image to specify the mask or enter values for the parameters of the mask
    print('Image A has width '+str(width)+', height '+str(height))
    while 1:
         mode = raw_input('Enter "1" to pick points to generate the mask, enter "2" to manually enter the parameters of the mask: ')
         if mode == '1' or mode == '2':
             break
    # use the pickPoints function from cvk2 library to get four vertices of the mask
    if mode == '1':
        datafile = 'blend.txt'
        print('Please pick the four vertices of the ellipse in the order of top, right, bottom, left by right clicking the image.')
        points = pickPoints('A', A, datafile)
        print('got points =\n', points)

        if len(points) != 4:
            print('Invalid number of points!')
            sys.exit(1)
        top = points[0][0]
        right = points[1][0]
        bottom = points[2][0]
        left = points[3][0]
        mean = (top+right+bottom+left)/4
        cx = int(mean[0])
        cy = int(mean[1])
        ellipse_width = int(max(distance(mean, left), distance(mean, right)))
        ellipse_height = int(max(distance(mean, top), distance(mean, bottom)))
        angle_rad = numpy.arccos((bottom[1]-top[1])/distance(bottom, top))
        if top[0]-bottom[0]>0:
            angle = angle_rad/numpy.pi*180
        else:
            angle = -angle_rad/numpy.pi*180

    # prompt user to enter the parameters of the ellipse
    else:
        cv2.namedWindow('A')
        cv2.imshow('A', A)
        cv2.namedWindow('B')
        cv2.imshow('B', B)
        while fixKeyCode(cv2.waitKey(15)) < 0:
            pass

        # prompt user for parameters of the alpha mask
        cx = int(raw_input('Enter the x-coord of the mask(-1 for default: center of the image): '))
        cy = int(raw_input('Enter the y-coord of the mask(-1 for default: center of the image): '))
        ellipse_width = int(raw_input('Enter the half-width of the ellipse(-1 for default: 1/4 of the image width): '))
        ellipse_height = int(raw_input('Enter the half-height of the ellipse(-1 for default: 1/4 of the image height): '))
        angle = int(raw_input('Enter the angle of the ellipse: '))

        if cx == -1:
            cx = width/2
        if cy == -1:
            cy = height/2
        if ellipse_width == -1:
            ellipse_width = width/4
        if ellipse_height == -1:
            ellipse_height = height/4

    sigma = int(raw_input('Enter the sigma of the Gaussian blur: '))
    ksize = int(raw_input('Enter the size of the Gaussian kernel: '))

    # generate alpha mask based on the specified parameters
    mask = numpy.zeros((height, width), dtype=numpy.uint8)
    cv2.ellipse(mask, (cx, cy), (ellipse_width, ellipse_height), angle, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    mask_blurred = cv2.GaussianBlur(mask, (ksize,ksize), sigma)
    alpha = mask_blurred.astype(numpy.float32) / 255.0

    cv2.namedWindow('alpha')
    cv2.imshow('alpha', alpha)
    cv2.imwrite('result/mask.jpg', alpha*255)

    # directly alpha blend the two images
    blend = alpha_blend(A, B, alpha)
    blend_clip = numpy.clip(blend, 0, 255)
    blend_8 = numpy.array(blend_clip, dtype = 'uint8')
    cv2.namedWindow('blend')
    cv2.imshow('blend', blend_8)
    cv2.imwrite('result/blend.jpg', blend_8)

    # pyramid alpha blend the two images
    pyramid_blend = LP_alpha_blend(A, B, alpha)
    pyramid_blend_clip = numpy.clip(pyramid_blend, 0, 255)
    pyramid_blend_8 = numpy.array(pyramid_blend_clip, dtype = 'uint8')
    cv2.namedWindow('pyramid blend')
    cv2.imshow('pyramid blend', pyramid_blend_8)
    cv2.imwrite('result/pyramid_blend.jpg', pyramid_blend_8)

# hybrid functionality
elif sys.argv[1] == 'hybrid':
    if len(sys.argv) != 4:
        print('usage: {} blend IMAGE1 IMAGE2'.format(sys.argv[0]))
        sys.exit(1)

    filename1 = sys.argv[2]
    filename2 = sys.argv[3]
    A = cv2.imread(filename1)
    B = cv2.imread(filename2)

    B = align(A, B)

    # tested parameters that worked well
    k_a = 0.8
    k_b = 1.2
    sigma_a = 50
    sigma_b = 30
    size_a = 55
    size_b = 15

    # generate low-pass filter for image A and high-pass filter for image B
    A_low = cv2.GaussianBlur(A, (size_a,size_a), sigma_a).astype('float32')
    B_low = cv2.GaussianBlur(B, (size_b,size_b), sigma_b).astype('float32')
    B_high = B - B_low

    I = k_a * A_low + k_b * B_high

    # display the original images and save the new image
    I_clip = numpy.clip(I, 0, 255)
    I_8 = numpy.array(I_clip, dtype = 'uint8')
    cv2.namedWindow('A')
    cv2.imshow('A', A)
    cv2.namedWindow('B')
    cv2.imshow('B', B)
    cv2.namedWindow('hybrid')
    cv2.imshow('hybrid', I_8)
    cv2.imwrite('result/hybrid.jpg', I_8)

else:
    print('usage: {} pyramid IMAGE1'.format(sys.argv[0]))
    print('usage: {} blend IMAGE1 IMAGE2'.format(sys.argv[0]))
    print('usage: {} hybrid IMAGE1 IMAGE2'.format(sys.argv[0]))

    sys.exit(1)


while fixKeyCode(cv2.waitKey(15)) < 0:
    pass
