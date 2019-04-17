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


def get_cnt(img):
    if img is not None:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        print(contours[0].shape)
        print(hierarchy)
        return contours


def feature():
    src = util.load_img('img/s.png')
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(src, 127, 255, cv.THRESH_BINARY)

    # cv.findContours（）函数中有三个参数，第一个是源图像，第二个是轮廓检索模式，第三个是轮廓近似方法。它输出轮廓和层次结构
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 第一个参数是源图像，第二个参数是轮廓，第三个参数是轮廓索引 -1是所有轮廓 其余参数是颜色，厚度
    # res = cv.drawContours(src, contours, -1, (0, 255, 0), 3)
    # util.show(res)

    height = src.shape[0]
    width = src.shape[0]

    out = util.blank_img(width * 1.5, height, (255, 255, 255))
    # util.show(img)

    for index, cnt in enumerate(contours):
        # 画轮廓
        cv.drawContours(out, contours, index, (0, 0, 255), 1)
        # 质心坐标
        M = cv.moments(cnt)
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        cv.circle(out, (int(cx), int(cy)), 2, (255, 0, 0), -1)
        # 轮廓面积 M['m00']
        area = cv.contourArea(cnt)
        # 轮廓周长   轮廓是否闭合True
        perimeter = cv.arcLength(cnt, True)

        print('({},{}) 面积={} 周长={}'.format(cx, cy, area, perimeter))

    util.show(out)


def approx():
    img = util.load_img('img/t.png')
    height = img.shape[0]
    width = img.shape[0]

    out = util.blank_img(width * 1.5, height, (255, 255, 255))
    contours = get_cnt(img)
    # a = cv.drawContours(out, contours, -1, (255, 0, 0), 5)

    epsilon = 1 * cv.arcLength(contours[0], True)
    approx = cv.approxPolyDP(contours[0], epsilon, True)
    cv.drawContours(out, [approx], -1, (0, 0, 255), 5)
    util.show(out)


def hull_():
    """
    类似于轮廓近似

    凸壳 在多维空间中有一群散佈各处的点 凸包 是包覆这群点的所有外壳当中，表面积暨容积最小的一个外壳，而最小的外壳一定是凸的

    凸： 图形内任意两点的连线不会经过图形外部

    """
    img = util.load_img('img/horse.png')
    convex = util.load_img('img/horse1.png')
    height = img.shape[0]
    width = img.shape[0]

    out = util.blank_img(width * 1.5, height, (255, 255, 255))
    # 求轮廓
    contours = get_cnt(img)
    convex_cnt = get_cnt(convex)[0]

    # 求凸包
    cv.drawContours(out, contours, -1, (255, 0, 0), 5)
    cnt = contours[0]
    hull = cv.convexHull(cnt)

    # 检测一个曲线是不是凸的
    print(cv.isContourConvex(cnt))

    cv.drawContours(out, [hull], -1, (0, 0, 255), 5)
    util.show(out)


def rect():
    img = util.load_img('img/rect.png')
    contours = get_cnt(img)

    height = img.shape[0]
    width = img.shape[0]

    # out = util.blank_img(width * 1.5, height, (255, 255, 255))
    out = img
    # out = img
    for index, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)
        # 原型轮廓 红
        cv.drawContours(out, contours, index, (0, 0, 255), 3)

        # 直边矩形 绿
        cv.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 最小面积绘制边界矩形 蓝
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box_ = np.int0(box)
        cv.drawContours(out, [box_], 0, util.rgb2bgr((30, 144, 255)), 2)

        # 最小封闭圈 橙
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center, radius = (int(x), int(y)), int(radius)
        cv.circle(out, center, radius, util.rgb2bgr((255, 140, 0)), 2)

        # 拟合椭圆 蓝
        ellipse = cv.fitEllipse(cnt)
        cv.ellipse(out, ellipse, util.rgb2bgr((135, 206, 250)), 2)
        print(index)

        # # 线 黑
        # rows, cols = out.shape[:2]
        # [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
        # lefty = int((-x * vy / vx) + y)
        # righty = int(((cols - x) * vy / vx) + y)
        # print((cols - 1, righty), (0, lefty))
        # cv.line(out, (cols - 1, righty), (0, lefty), (0, 0, 0), 2)

        # util.show(out)

    util.show(out)


def more():
    img = util.load_img('img/star.png')
    contours = get_cnt(img)
    cnt = contours[0]
    # 凸包时传递returnPoints = False，以便找到凸起缺陷
    hull = cv.convexHull(cnt, returnPoints=False)
    defects = cv.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        # 起点，终点，最远点，到最远点的近似距离
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv.line(img, start, end, [0, 255, 0], 2)
        cv.circle(img, far, 5, [0, 0, 255], -1)

    util.show(img)



more()