import cv2 as cv
from skimage.color import rgb2gray
import numpy as np
from skimage import filters
from scipy import signal
from skimage.segmentation import watershed
from matplotlib import pyplot


data = cv.VideoCapture('car.mp4')


kernel = np.ones((5,5), np.uint8)
kernel1 = np.ones((10,10), np.uint8)
ret, frame1 = data.read()
while data.isOpened():
    try:
        ret, frame2 = data.read()
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    except cv.error as e:
        break
    gray1 = cv.blur(gray1, (7, 7))
    gray2 = cv.blur(gray2, (7, 7))

    edge = cv.Laplacian(gray2, -1)
    # gray = np.uint8(gray)
    diff1 = cv.absdiff(gray2, gray1)
    bin1 = np.zeros(np.shape(diff1), dtype='uint8')
    bin1[diff1 >= np.max(diff1)/1.2] = 255
    # pyplot.imshow(diff1)
    # pyplot.show()
    # ret, bin1 = cv.threshold(diff1, 0, 255  , cv.THRESH_OTSU)
    ret, edge_bin = cv.threshold(edge, 0, 1, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    erosion1 = cv.dilate(bin1, kernel, iterations=2)
    erosion1 = cv.erode(erosion1, kernel, iterations=1)
    morpho = cv.erode(edge_bin, kernel, iterations=20)
    dummy = morpho.copy()
    # erosion1 = cv.dilate(erosion1, kernel, iterations=3)
    ret, markers = cv.connectedComponents(erosion1)
    markers1 = morpho + markers
    markers1 = np.int32(markers1)
    dummy1 = markers1.copy()
    ganesh = cv.watershed(frame2, markers1)
    ganesh = np.uint8(ganesh)
    ganesh1 = frame2.copy()
    ganesh1[:, :, 1] = cv.add(ganesh1[:, :, 1], cv.dilate(ganesh,kernel,iterations=1))
    # pyplot.imshow(ganesh1)
    # pyplot.show()
    cv.imshow('ganesh',diff1)
    cv.imshow('ganesh2',ganesh1)

    cv.waitKey(50)
    frame1 = frame2.copy()



    # for cnt in contours:
        # rect = cv.minAreaRect(cnt)
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # ganesh1 = cv.drawContours(ganesh, [box], 0, (0, 255, 0), 20)
        # print(cnt)
    # pyplot.subplot(221)
    # pyplot.imshow(markers)
    # pyplot.subplot(222)
    # pyplot.imshow(morpho)
    # pyplot.subplot(223)
    # pyplot.imshow(dummy1)
    # pyplot.subplot(224)
    # pyplot.imshow(ganesh)
    # pyplot.show()