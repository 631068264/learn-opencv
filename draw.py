#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2019-04-08 21:28
@annotation = ''
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import util


def draw():
    # 画图窗口大小设置为512*512 通常参数，可选1,3,4
    img = np.zeros((512, 512, 4), np.uint8)

    # 画直线，参数为绘图窗口，起点，终点，颜色，粗细，连通性（可选4联通或8联通）
    img = cv.line(img, (100, 0), (511, 511), (255, 0, 0), 5, 8)
    # 画矩形，参数为窗口，左上角坐标，右下角坐标，颜色，粗度
    img = cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    # 画圆函数，参数为窗口，圆心，半径，颜色，粗度（如果粗度为-1标示实心）,连通性
    img = cv.circle(img, (447, 63), 50, (0, 0, 255), 1, 8)
    # 画椭圆函数，参数为窗口，椭圆中心，椭圆长轴短轴，椭圆逆时针旋转角度，椭圆起绘制起始角度，椭圆绘制结束角度，颜色，粗度，连通性
    img = cv.ellipse(img, (256, 256), (100, 50), 100, 36, 360, (0, 255, 255), 1, 8)

    # 画多边形
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    # 转换成三维数组，表示每一维里有一个点坐标
    pts = pts.reshape((-1, 1, 2))
    # 画多边形函数，参数为窗口，坐标，是否封闭，颜色
    img = cv.polylines(img, [pts], True, (0, 255, 255))

    # 字 参数为窗口，字符，字体位(左下角),字体类型(查询cv2.putText()函数的文档来查看字体),字体大小,常规参数，类似颜色，粗细，线条类型等等
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

    util.show(img)


drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1


def mouse_draw():
    """双击画圆"""

    def draw_circle(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(img, (x, y), 100, (255, 0, 0), -1)

    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode
        # print(ix, iy, drawing, mode)
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                if mode:
                    cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
                else:
                    cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            if mode:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_circle)

    while True:
        cv.imshow('image', img)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()


def bar_grb():
    def nothing(x):
        pass

    # Create a black image, a window
    img = np.zeros((300, 512, 3), np.uint8)
    cv.namedWindow('image')

    # create trackbars for color change
    cv.createTrackbar('R', 'image', 0, 255, nothing)
    cv.createTrackbar('G', 'image', 0, 255, nothing)
    cv.createTrackbar('B', 'image', 0, 255, nothing)
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv.createTrackbar(switch, 'image', 0, 1, nothing)

    while True:
        cv.imshow('image', img)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        # get current positions of four trackbars
        r = cv.getTrackbarPos('R', 'image')
        g = cv.getTrackbarPos('G', 'image')
        b = cv.getTrackbarPos('B', 'image')
        s = cv.getTrackbarPos(switch, 'image')
        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

    cv.destroyAllWindows()


def his():
    """
    # cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    """
    # hist = cv.calcHist([img], [0], None, [256], [0, 256])
    #
    # hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    # mask = np.zeros(img.shape[:2], np.uint8)
    # mask[100:300, 100:400] = 255
    # masked_img = cv.bitwise_and(img, img, mask=mask)

    # gray
    img = util.load_img('img/home.jpg', 0)
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()

    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv.bitwise_and(img, img, mask=mask)

    hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0, 256])
    plt.show()

    # color
    img = util.load_img('img/home.jpg')
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        #
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def contrast():
    # img = util.load_img('img/contrast.png', 0)
    # equ = cv.equalizeHist(img)
    # res = np.hstack((img, equ))  # stacking images side-by-side
    # util.show(res)

    img = util.load_img('img/tsukuba.png', 0)
    # 直方图均衡
    equ = cv.equalizeHist(img)
    """
    自适应直方图均衡解决 增加背景噪音的对比度 增加局部对比度
    
    图像被分成称为tile（8x8）的小块.每个小块进行直方图均衡, 加入对比度限制防止噪音被放大
    超过限制，像素剪切并均匀分布到其他区间，均衡后，为了去除图块边框中的瑕疵，应用双线性插值。
    """
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    res = np.hstack((equ, cl1))
    util.show(res)


def search_roi():
    roi = util.load_img('img/roi.png')
    roi_hsv = util.hsv(roi)
    tar = util.load_img('img/tar.png')
    tar_hsv = util.hsv(tar)

    # 计算目标直方图 颜色直方图优于灰度直方图
    roihist = cv.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 对象直方图进行normalize cv.calcBackProject返回概率图像
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([tar_hsv], [0, 1], roihist, [0, 180, 0, 256], 1)
    # 与disc kernel卷积
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cv.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    ret, thresh = cv.threshold(dst, 50, 255, 0)
    # 使用merge变成通道图像
    thresh = cv.merge((thresh, thresh, thresh))
    # 蒙板
    res = cv.bitwise_and(tar, thresh)

    res = np.hstack((tar, thresh, res))
    util.show(res)


def temp():
    """
    模板匹配
        较大图像中搜索和查找模板图像位置的方法

    与2D卷积一样 模板图像在输入图像上滑动（类似窗口），在每一个位置对模板图像和输入图像的窗口区域进行匹配。 与直方图的反向投影类似。

    输入图像大小是W×H，模板大小是w×h，输出结果的大小(W-w+1,H-h+1)。
    得到此结果后可以使用函数cv2.minMaxLoc()来找到其中的最小值和最大值的位置。第一个值为矩形左上角的位置，(w,h)是模板矩形的宽度和高度。矩形就是模板区域。

    """
    roi = util.load_img('img/roi.png', 0)
    w, h = roi.shape[::-1]
    tar = util.load_img('img/tar.png', 0)
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    for meth in methods:
        img = roi.copy()
        method = eval(meth)

        res = cv.matchTemplate(img, tar, method)
        # 只匹配一个对象
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            # 最小值会给出最佳匹配
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()


def more_temp():
    """匹配多个对象"""
    img_rgb = util.load_img('img/mario.png')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread('img/mario_coin.png', 0)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    print(res.shape)
    threshold = 0.8
    # 注意 行列 高宽
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    util.show(img_rgb)


def water():
    """
    使用距离变换和分水岭来分割相互接触的物体

    靠近对象中心的区域是前景，离对象远的区域是背景，不确定的区域是边界。

    物体没有相互接触/只求前景 可用侵蚀消除了边界像素

    到距离变换并应用适当的阈值  膨胀操作会将对象边界延伸到背景，确保background区域只有background

    边界 = 能否确认是否是背景的区域  - 确定是前景的区域
    """
    img = util.load_img('img/coins.png')
    gray = util.gray(img)

    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    util.show(thresh)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5) # 计算每个像素离最近0像素的距离
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling 用0标记图像的背景 其他对象用从1开始的整数标记
    ret, markers = cv.connectedComponents(sure_fg)

    """
    我们知道，如果背景标记为0，分水岭会将其视为未知区域。所以我们想用不同的整数来标记它。相反，我们将标记由未知定义的未知区域，为0。
    """
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    # 修改标记图像。边界区域将标记为-1
    img[markers == -1] = [255, 0, 0]

    util.show(img,is_seq=True)


def cut():

    img = util.load_img('img/messi5.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()


cut()