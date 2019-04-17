#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2019-04-08 21:28
@annotation = ''
"""
import cv2 as cv
import numpy as np

import util

"""
BGR和灰度图的转换使用 cv.COLOR_BGR2GRAY
 
BGR和HSV的转换使用 cv.COLOR_BGR2HSV
H表示色彩/色度，取值范围 [0，179]
S表示饱和度，取值范围 [0，255]
V表示亮度，取值范围 [0，255]
"""

img = util.load_img('img/messi5.jpg')
img2gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

util.show(img2gray)

cap = cv.VideoCapture(0)
while (1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    """
    BGR2HSV
    
    green = np.uint8([[[0,255,0 ]]])
    hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    
    使用[H-10, 100,100] and [H+10, 255, 255] 做阈值上下限
    """
    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # 设定取值范围
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
