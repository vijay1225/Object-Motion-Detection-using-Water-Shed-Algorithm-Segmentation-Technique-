import cv2 as cv
from skimage.color import rgb2gray
import numpy as np
from skimage import filters
from scipy import signal
from skimage.segmentation import watershed

i = 1
data = cv.VideoCapture('car.mp4')

sobel_x = np.reshape([1, 0, -1, 2, 0, -2, 1, 0, -1], [3, 3])
sobel_y = np.reshape([1, 2, 1, 0, 0, 0, -1, -2, -1], [3, 3])

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
    kernel = np.ones((3,3), np.uint8)
    erosion1 = cv.erode(diff1, kernel, iterations=1)
    erosion2 = cv.erode(diff2, kernel, iterations=1)

    diff = (erosion1 * erosion2)

    frame1 = frame2
    cv.imshow('ganesh',diff1)
    cv.imshow('ganesh2',edge_mag)
    cv.waitKey(100)