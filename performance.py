#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2019-04-08 21:28
@annotation = ''
"""
import cv2 as cv

# 时钟周期数
e1 = cv.getTickCount()
# your code execution
e2 = cv.getTickCount()

# 时间 = 时钟周期数/时钟周期的频率
time = (e2 - e1)/ cv.getTickFrequency()

print(time)