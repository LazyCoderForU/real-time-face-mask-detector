# -----------------------------------------
# Importing essential libraries
# -----------------------------------------
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

# -----------------------------------------
# Function to detect faces and predict mask presence
# -----------------------------------------
def detect_and_predict_mask(frame, face_nn, mask_nn):
    # Get the dimensions of the frame
    (h, w) = frame.shape[:2]
    
    # Create a blob from the input frame to feed into face detection model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    # Pass the blob through the face detection network
    face_nn.setInput(blob)
    detections = face_nn.forward()
    
    # Lists to store face ROIs, their locations, and predictions
    Array_faces = []
    Array_locs = []
    Array_preds = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure bounding boxes fall within frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            # Append the face and its location
            Array_faces.append(face)
            Array_locs.append((startX, startY, endX, endY))
    
    # Only make predictions if at least one face was detected
    if len(Array_faces) > 0:
        Array_faces = np.array(Array_faces, dtype="float32")
        Array_preds = mask_nn.predict(Array_faces, batch_size=32)
    
    # Return face locations and their corresponding predictions
    return (Array_locs, Array_preds)

# -----------------------------------------
# Load the pre-trained face detection and mask detection models
# -----------------------------------------

# Paths to face detection model files
t = "architecture.txt"   # path to the prototxt file (model architecture)
w = "weights.caffemodel"  # path to the caffemodel file (pre-trained weights)

# Load models
face_nn = cv2.dnn.readNet(t, w)           # Load face detection model
mask_nn = load_model("mask_detector.h5")  # Load mask detection model

# -----------------------------------------
# Start the video stream
# -----------------------------------------
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Give the camera some time to warm up

# -----------------------------------------
# Real-time frame reading and processing loop
# -----------------------------------------
while True:
    # Read a frame from the video stream
    frame = vs.read()
    
    # Resize the frame to 400px width for faster processing
    frame = imutils.resize(frame, width=400)
    
    # Detect faces and predict mask presence
    (Array_locs, Array_preds) = detect_and_predict_mask(frame, face_nn, mask_nn)
    
    # Loop over the detected face locations and their corresponding predictions
    for (box, pred) in zip(Array_locs, Array_preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        # Determine the class label and bounding box color
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)  # Green
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red
        
        # Include probability/confidence in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # Show the output frame
    cv2.imshow("Frame", frame)
    
    # If the 's' key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        break

# -----------------------------------------
# Cleanup: close windows and stop video stream
# -----------------------------------------
cv2.destroyAllWindows()
vs.stop()
