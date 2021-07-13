import cv2 as cv
from skimage.color import rgb2gray
import numpy as np
from skimage import filters
from scipy import signal
from skimage.segmentation import watershed
from matplotlib import pyplot


data = cv.VideoCapture('jerry.mp4')


kernel = np.ones((5,5), np.uint8)
ret, frame1 = data.read()
while data.isOpened():
    ret, frame2 = data.read()
    ret, frame3 = data.read()

    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    gray3= cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
    #
    # gray1 = cv.blur(gray11,(7, 7))
    # gray2 = cv.blur(gray22, (7, 7))
    # gray3 = cv.blur(gray33, (7, 7))
    #
    edge = cv.Laplacian(gray2, -1)
    # gray = np.uint8(gray)
    diff1 = cv.absdiff(gray2, gray1)
    diff2 = cv.absdiff(gray2, gray3)
    diff = cv.absdiff(diff1, diff2)
    ret, bin1 = cv.threshold(diff1, 0, 255, cv.THRESH_OTSU)
    ret, bin2 = cv.threshold(diff2, 0, 255, cv.THRESH_OTSU)
    ret, edge_bin = cv.threshold(edge, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    erosion1 = cv.dilate(bin1, kernel, iterations=2)
    erosion1 = cv.erode(erosion1, kernel, iterations=5)
    morpho = cv.erode(edge_bin, kernel, iterations=4)
    dummy = morpho.copy()
    morpho[morpho == 255] = 1
    # erosion1 = cv.dilate(erosion1, kernel, iterations=3)
    ret, markers = cv.connectedComponents(erosion1)

    dummy2 = np.uint8(markers.copy())
    markers1 = markers + morpho
    ganesh = cv.watershed(frame2, markers1)
    ganesh = np.uint8(ganesh)
    # pyplot.imshow(ganesh)
    # pyplot.show()
    cv.imshow('ganesh',frame2)
    cv.imshow('ganesh2',ganesh)

    cv.waitKey(100)
    frame1 = frame2.copy()