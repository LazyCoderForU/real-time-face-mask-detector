import os
import cv2
import time
import keras
import imutils
import numpy as np

from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream

def detect_and_predict_mask(frame, face_nn, mask_nn):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    face_nn.setInput(blob)
    detections = face_nn.forward()
    
    Array_faces = []
    Array_locs = []
    Array_preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            Array_faces.append(face)
            Array_locs.append((startX, startY, endX, endY))
    
    if len(Array_faces) > 0:
        Array_faces = np.array(Array_faces, dtype="float32")
        Array_preds = mask_nn.predict(Array_faces, batch_size=32)
    
    return (Array_locs, Array_preds)

# Load models
face_nn = cv2.dnn.readNet("architecture.txt", "weights.caffemodel")
mask_nn = load_model("mask_detector.h5")

# Start video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Main loop
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    (Array_locs, Array_preds) = detect_and_predict_mask(frame, face_nn, mask_nn)
    
    for (box, pred) in zip(Array_locs, Array_preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)
        
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        break

cv2.destroyAllWindows()
vs.stop()
