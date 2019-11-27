from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2


def config_params(ap):
    ap.add_argument('-m',
                    '--model',
                    type=str,
                    default='ln',
                    help="liveness's model filename without extension")
    ap.add_argument('-d',
                    '--detector',
                    type=str,
                    default='opencv_face_detector',
                    help="face_detector's model filename without extension")
    ap.add_argument('-c',
                    '--confidence',
                    type=float,
                    default=0.8,
                    help='minimum probability accdepting detected faces')
    args = vars(ap.parse_args())

    # load our serialized face detector from disk
    print('[INFO] loading face detector...')
    face_detector_name = args['detector']
    protoPath = 'face_detector/' + face_detector_name + '.prototxt'
    modelPath = 'face_detector/' + face_detector_name + '.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load the liveness detector model and label encoder from disk
    print('[INFO] loading liveness detector...')
    model_name = args['model']
    model_def = 'spoofing/' + model_name + '.model'
    label_enc = 'spoofing/' + model_name + '.pickle'
    model = load_model(model_def)
    le = pickle.loads(open(label_enc, 'rb').read())

    # set minimum confidence for face photo
    min_conf = args['confidence']

    return net, model, le, min_conf


def analyze(frame, conf, net, model, le):
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    # print(detections[0, 0, :, 2])

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > conf:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (iniX, iniY, endX, endY) = box.astype('int')

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            iniX = max(0, iniX)
            iniY = max(0, iniY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preprocess it in the exact
            # same manner as our training data
            face = frame[iniY:endY, iniX:endX]
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
            cv2.putText(frame, label, (iniX + 2, iniY + 15), f, 0.5, color, 2)
            cv2.rectangle(frame, (iniX, iniY), (endX, endY), color, 2)
    return frame
