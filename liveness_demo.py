"""
Created on Mon Mar 25 21:29:05 2019
@author: bhaum
Modifyed:
Miguel Pari Soto
"""

# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import FPS
from pathlib import Path
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-m',
                '--model',
                type=str,
                default='liveness.model',
                help='path to trained model')
ap.add_argument('-l',
                '--le',
                type=str,
                default='le.pickle',
                help='path to label encoder')
ap.add_argument('-d',
                '--detector',
                type=str,
                default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument('-c',
                '--confidence',
                type=float,
                default=0.5,
                help='minimum probability accdepting detected faces')
args = vars(ap.parse_args())

# load our serialized face detector from disk
print('[INFO] loading face detector...')
protoPath = str(Path(args['detector']) / 'deploy.prototxt')
modelPath = str(
    Path(args['detector']) / 'res10_300x300_ssd_iter_140000.caffemodel')
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print('[INFO] loading liveness detector...')
model = load_model(args['model'])
le = pickle.loads(open(args['le'], 'rb').read())

# initialize the video stream and allow the camera sensor to warmup
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    e1 = cv2.getTickCount()
    frame = imutils.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args['confidence']:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            face = frame[startY:endY, startX:endX]
            # print(face.shape)
            if face.shape[0] < 64 or face.shape[1] < 64:
                continue
            face = cv2.resize(face, (64, 64))
            face = face.astype('float') / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # pass the face ROI through the trained liveness detector
            # model to determine if the face is 'real' or 'fake'
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            if str(label) == str('real'):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # draw the label and bounding box on the frame
            label = f'{label}: {preds[j]:.4f}'
            f = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (startX, startY - 10), f, 0.5, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame and wait for a key press
    cv2.imshow('Frame', frame)
    e2 = cv2.getTickCount()
    time1 = (e2 - e1) / cv2.getTickFrequency()
    print(f'[INFO] elasped time: {time1:.2f}')
    print(f'[INFO] approx. FPS: {1/time1:.2f}')
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break
fps.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
