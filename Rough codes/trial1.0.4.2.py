import cv2 as cv
from skimage.color import rgb2gray
import numpy as np
from skimage import filters
from scipy import signal
from skimage.segmentation import watershed

i = 1
data = cv.VideoCapture('jerry.mp4')

sobel_x = np.reshape([1, 0, -1, 2, 0, -2, 1, 0, -1], [3, 3])
sobel_y = np.reshape([1, 2, 1, 0, 0, 0, -1, -2, -1], [3, 3])
kernel = np.ones((5,5), np.uint8)
ret, frame1 = data.read()
while data.isOpened():
    ret, frame2 = data.read()
    ret, frame3 = data.read()

    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    gray3 = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)



    diff1 = np.abs(gray2 - gray1)
    diff2 = np.abs(gray3 - gray2)

    edge_x = signal.convolve2d(diff1, sobel_x, boundary='symm', mode='same')
    edge_y = signal.convolve2d(diff1, sobel_y, boundary='symm', mode='same')
    edge_mag = np.sqrt((edge_x ** 2) + (edge_y ** 2))

    erosion1 = cv.dilate(edge_mag, kernel, iterations=1)
    erosion2 = cv.erode(erosion1, kernel, iterations=10)

    diff = (erosion1 * erosion2)

    frame1 = frame2
    cv.imshow('ganesh',frame2)
    cv.imshow('ganesh2',edge_mag)
    cv.waitKey(50)