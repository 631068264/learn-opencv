#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2019-04-08 21:28
@annotation = ''
"""
import cv2 as cv


def cap0():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    """
    cap.get/set(proid) proid是0到18的数字每个数组表示一个视频的属性
    https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    
    """
    # print('width {:.2f}'.format(cap.get(3)))
    # print('height {:.2f}'.format(cap.get(4)))

    # 不要把set写到循环里，会闪瞎
    cap.set(3, 320)
    cap.set(4, 240)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def vi(path='img/vtest.avi'):
    cap = cv.VideoCapture(path)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def cap_save():
    cap = cv.VideoCapture(0)
    # Define the codec and create VideoWriter object
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    # http://www.ntta.szm.com/Tutors/FourCC.htm#QT4CC
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, size, )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv.flip(frame, 0)
        # write the flipped frame
        out.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()
