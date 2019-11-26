#!/usr/bin/env python
"""
Created on Mon Mar 25 21:29:05 2019
@author: bhaum
Modified:
Miguel Pari Soto
"""

# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

from imutils.video import VideoStream
from imutils.video import FPS
from analyze_frame import analyze, config_params
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()  # <===
(net, model, le, min_conf) = config_params(ap)  # <===

# initialize the video stream and allow the camera sensor to warmup
print('[INFO] starting video stream...')
cap = cv2.VideoCapture(0)
time.sleep(2.0)

fps = FPS().start()
# loop over the frames from the video stream
while (cap.isOpened()):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = cap.read()[1]
    e1 = cv2.getTickCount()
    frame = imutils.resize(frame, width=600)
    frame = analyze(frame, min_conf, net, model, le)  # <===
    # show the output frame and wait for a key press
    cv2.imshow('Frame', frame)
    e2 = cv2.getTickCount()
    time1 = (e2 - e1) / cv2.getTickFrequency()
    print(f'[INFO] elapsed time ~ {time1:.2f}, FPS ~ {1/time1:.2f}')
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break
fps.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
