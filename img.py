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


def img():
    """
    cv.IMREAD_COLOR：加载彩色图像。任何图像的透明度都将被忽略。这是默认标志。
    cv.IMREAD_GRAYSCALE：以灰度模式加载图像
    cv.IMREAD_UNCHANGED：加载图像，包括alpha通道

    整数1,0或-1
    """
    img_mode = cv.IMREAD_GRAYSCALE
    img = cv.imread('img/logo.png', img_mode)
    util.plt_gray_img(img)
    # print(img)

    cv.imwrite('xx.jpg', img)


def img_ctl():
    """一个BGR图像"""
    img = util.load_img('img/messi5.jpg')
    # 属性
    print(img.shape)  # 行数、列数、[通道数]
    print(img.size)  # 总像素数
    print(img.dtype)  # 数据类型

    # 选择数组的区域 前5行和后3列
    px = img[50, 50]
    print(px)
    px = img[50, 50, 0]  # 获取blue值
    print(px)
    px = img[50, 50, 1]  # 获取green值
    print(px)
    px = img[50, 50, 2]  # 获取red值
    print(px)

    img[50, 50] = [255, 255, 255]
    print(img[50, 50])

    # copy
    ball = img[280:340, 330:390]
    img[273:333, 100:160] = ball
    util.save_img('out.jpg', img)
    # 分割和合并图像通道 split 耗时 没什么必要可以用numpy的index
    b, g, r = cv.split(img)
    img2 = cv.merge([r, g, b])

    # 单个像素访问item() itemset()
    print('\n', img[10, 10])
    sca = img.item(10, 10, 0)  # 获取bgr值
    print(sca)
    sca = img.item(10, 10, 1)
    print(sca)
    sca = img.item(10, 10, 2)
    print(sca)


def img_cal():
    """图像运算"""
    x = np.uint8([250])
    y = np.uint8([10])
    """
    Both images should be of same depth and type, (shape and 图片后缀)
    or second image can just be a scalar value.

    OpenCV添加是饱和操作，而Numpy添加是模运算。
    """
    print(cv.add(x, y))  # 250+10 = 260 => 255
    print(x + y)  # 250+10 = 260 % 256(2^8) = 4
    img1 = util.load_img('img/ml.png')
    img2 = util.load_img('img/opencv-logo.png')
    print(img1.shape, img2.shape)
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    dst = cv.add(img1, img2)
    util.show(dst)

    """
    图像混合 按比例混合起来，有不同的权重 修改透明度
    dst=α⋅img1+β⋅img2+γ
    """
    img1 = util.load_img('img/ml.png')
    img2 = util.load_img('img/opencv-logo.png')
    print(img1.shape, img2.shape)
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    assert img1.shape == img2.shape
    print(img1.shape, img2.shape)
    # 有不同的权重
    dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
    util.show(dst)

    """
    按位AND，OR，NOT和XOR运算 添加两个图像，它将改变颜色。如果我混合它，我会得到一个透明的效果
    
    更改图像的特定区域 Region Of Interest ROI 感兴趣区域
    """
    img1 = util.load_img('img/messi5.jpg')
    img2 = util.load_img('img/opencv-logo-white.png')

    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    """
    what is mask 掩膜
    
    0 黑 255 白
    
    与目标图像做mask操作 目标图像扣走mask中黑色轮廓部分，保留白色区域 => 保留ROI,其他区域为0
    
    通常mask之前
    先转成灰度图像
    二值化、反二值化
    
    对自己做mask 可以抠图
    提取ROI
    特征提取
    
    """
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)

    util.save_img('img1_bg.png', img1_bg)
    util.save_img('img2_fg.png', img2_fg)

    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    util.show(mask_inv, img1_bg, mask, img2_fg)
    util.show(img1)
    # util.show(mask_inv)


def trans():
    """
    translation, rotation, affine transformation

    img.shape height width channel

    x 横坐标 y 纵坐标
    resize( (width,height),fx,fy (因子) )
    cv.INTER_AREA for 缩小
    cv.INTER_LINEAR for 放大 default
    cv.INTER_CUBIC slow

    :return:
    """
    # img = util.load_img('img/messi5.jpg')
    # height, width = img.shape[:2]
    # print(img.shape)
    # res = cv.resize(img, None, fx=3, fy=2, interpolation=cv.INTER_LINEAR)
    # res = cv.resize(img, (int(0.5 * width), int(0.5 * height)), interpolation=cv.INTER_AREA)
    #
    # print(res.shape)
    # util.show(res)

    """
    平移 （100，50）
    """
    img = util.load_img('img/messi5.jpg', 0)
    print(img.shape)
    rows, cols = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(img, M, (cols, rows))
    print(dst.shape)
    util.show(dst)
    """
    旋转 90度
    """
    rows, cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    # center, angle, scale
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    print(dst.shape)
    util.show(dst)


def threshold():
    """
    二值化
    """

    img = util.load_img('img/gradient.png', 0)
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    """
    自适应阈值
    
    照明条件
    
    算法基于其周围的小区域确定像素的阈值。
    因此，我们为同一图像的不同区域获得不同的阈值，这为具有不同照明的图像提供了更好的结果。
    
    cv.ADAPTIVE_THRESH_MEAN_C 阈值= 取邻近区域的平均值 - 常数C
    cv.ADAPTIVE_THRESH_MEAN_C 阈值= 邻近区域的高斯加权和 - 常数C
    adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst
    """
    img = util.load_img('img/sudoku.png', 0)
    util.show(img)
    # smoothes an image using the median filter 二值化效果更好
    img = cv.medianBlur(img, 5)
    util.show(img)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    """
    全局变量的阈值当中，使用一个比较模糊的值，那么怎么才能知道选择的值是好是坏？答案是实验和尝试法。
    一个双峰的图像（就是图像的直方图有两个峰值 可以大约的取到一个在两个峰值之间的值作为阈值。这就是Otsu二值化的方法
    """
    img = util.load_img('img/noisy1.png', 0)
    # global thresholding
    ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


def smooth():
    """
    图像过滤
    低通滤波LPF可以使图像去除噪声，高通滤波HPF可以找到图像的边缘。

    图像平滑
    内核卷积来实现图像模糊。它有助于消除噪音。实际上从图像中去除了高频内容（例如：噪声，边缘）。边缘会有点模糊。

    均值过滤
    调用blur()等效于调用将normalize=true的boxFilter().

    中位数
     cv.medianBlur() 内核区域下所有像素的中值，并用该中值替换中心元素

    双边过滤
    cv.bilateralFilter() 降低噪音方面非常有效，同时保持边缘清晰。但与其他过滤器相比，操作速度较慢。
    高斯滤波器采用像素周围的邻域并找到其高斯加权平均值

    :return:
    """
    img = util.load_img('img/opencv-logo-white.png')
    cv.blur(img, (5, 5))
    blur = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.medianBlur(img, 5)
    cv.blur(img, (5, 5))


def morphological():
    """形态学运算符是侵蚀和膨胀"""
    # img = util.load_img('img/j.png', 0)
    #
    # """
    # 侵蚀 Erosion 如果与卷积核对应的原图像像素值都是1，那么中心元素保持原值，否则为0
    # 有助于消除小的白噪声
    # """
    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv.erode(img, kernel, iterations=1)
    #
    # """
    # 扩张 Dilation 恰好与侵蚀相反 由于噪音消失了，它们不会再回来
    # """
    # dilation = cv.dilate(img, kernel, iterations=1)
    # """
    # 开运算  先腐蚀再膨胀，一般用来去除噪声
    # """
    # opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # """
    # 闭运算  先膨胀再腐蚀，一般用来填充黑色的小像素点
    # """
    # closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    # util.plt_gray_img(img, closing)
    """
    梯度滤波器或高通滤波器，Sobel，Scharr和Laplacian。
    """
    img = util.load_img('img/sudoku.png', 0)
    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()


def edge_detection():
    """
    Edge Detection

    Canny Edge Detection

    去除噪声 remove the noise in the image with a 5x5 Gaussian filter
    计算图像梯度 在水平与竖直方向上计算一阶导数，图像梯度方向和大小
    去除噪声 remove the noise in the image with a 5x5 Gaussian filter
    去除噪声 remove the noise in the image with a 5x5 Gaussian filter
    """
    img = util.load_img('img/messi5.jpg', 0)
    edges = cv.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def pyramid():
    # """
    # 有不同分辨率的相同图像
    #
    # 不同分辨率的图像就是图像金字塔（小分辨率的图像在顶部，大的在底部
    #
    # 低分辨率的图像由高分辨率的图像去除连续的行和列得到
    # """
    #
    # img = util.load_img('img/messi5.jpg')
    # # 分辨率减少 M×N的图像就变成了M/2×N/2的图像了，面积变为原来的四分之一
    # lower_reso = cv.pyrDown(img)
    # # 尺寸变大 分辨率不变
    # high_reso = cv.pyrUp(lower_reso)
    # util.show(img, lower_reso, high_reso)

    A = util.load_img('img/apple.jpg')
    B = util.load_img('img/orange.jpg')

    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        GE = cv.pyrUp(gpA[i])
        L = cv.subtract(gpA[i - 1], GE)
        lpA.append(L)
    util.show(*lpA)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        GE = cv.pyrUp(gpB[i])
        L = cv.subtract(gpB[i - 1], GE)
        lpB.append(L)
    util.show(*lpB)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
        LS.append(ls)
    util.show(*LS)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv.pyrUp(ls_)
        ls_ = cv.add(ls_, LS[i])
    # image with direct connecting each half
    real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))
    cv.imwrite('Pyramid_blending2.jpg', ls_)
    cv.imwrite('Direct_blending.jpg', real)


def hough():
    img = util.load_img('img/sudoku.png')
    img_gray = util.gray(img)
    edges = cv.Canny(img_gray, 100, 200)

    """cv.HoughLinesP"""
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    """cv.HoughLines"""
    # lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    # for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    """"""

    util.show(img)


hough()
