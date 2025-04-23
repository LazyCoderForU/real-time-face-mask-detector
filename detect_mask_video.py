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
#--------------------------------------------------------------------------------------
def detect_and_predict_mask(frame, face_nn, mask_nn):
    
    
	(h, w) = frame.shape[:2]    #getting first 2 value ie, height and weidth of frame 
 
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))  
 #dataset requires (104.0, 177.0, 123.0) mean rgb value to subtract from current frame for better prediction
 # creating large b input
 
	face_nn.setInput(blob)   # input type blog data (image)  to pretained face nn 
	detections = face_nn.forward()    #predicting the face boundaries and storing to var detections
	print(detections.shape) 
	Array_faces = []   
	Array_locs = []
	Array_preds = []









#2
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]          #for each box 
  
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












#3
	if len(Array_faces) > 0:
		Array_faces = np.array(Array_faces, dtype="float32")
		Array_preds = mask_nn.predict(Array_faces, batch_size=32)

	return (Array_locs, Array_preds)



#1
#--------------------------------------------------------------------------------------
def detect_and_predict_mask(frame, face_nn, mask_nn):
    
    
	(h, w) = frame.shape[:2]    #getting first 2 value ie, height and weidth of frame 
 
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))  
 #dataset requires (104.0, 177.0, 123.0) mean rgb value to subtract from current frame for better prediction
 
	face_nn.setInput(blob)
	detections = face_nn.forward()    #predicting the face is present in frame or not
	print(detections.shape)
	Array_faces = []   
	Array_locs = []
	Array_preds = []

#2
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

#3
	if len(Array_faces) > 0:
		Array_faces = np.array(Array_faces, dtype="float32")
		Array_preds = mask_nn.predict(Array_faces, batch_size=32)

	return (Array_locs, Array_preds)

#-------------------------------------------------------------------------------------

# prototxt_Path = "Face-Mask-Detection-master/face_detector/deploy.prototxt"
# weightsPath_caffemodel = "Face-Mask-Detection-master/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
#4




















t = "archive/architecture.txt"
w = "archive/weights.caffemodel"

face_nn = cv2.dnn.readNet(t, w)
mask_nn = load_model("mask_detector.h5")

# -------------------------------------------------------------------------------------------------------------------------------------

vs = VideoStream(src=0).start()

# we used while as with for loop or other kinds of loop it will not continously read frame till pressed q 



#5
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)  
	(Array_locs, Array_preds) = detect_and_predict_mask(frame, face_nn, mask_nn)

#6
	for (box, pred) in zip(Array_locs, Array_preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		if mask > withoutMask:
			label = "Mask"
		else:
			label = "No Mask"
	
  
		if label == "Mask":
			color = (0, 255, 0)
		else:
			color = (0, 0, 255)


		label = "{}= {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

#7
	cv2.imshow("Frame", frame)           # granting access 
	key = cv2.waitKey(1) & 0xFF

	if key == ord("s"):
		break



cv2.destroyAllWindows()
vs.stop() 

