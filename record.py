import os
import cv2
import random
import string
import math
from random import shuffle
import numpy as np
import datetime
from imutils.video import WebcamVideoStream
import cv2
import imutils
import autopy
from config import datasetImageSize, cascadePath, rawPath, screenSize
from datasetTools import save

#HAAR Cascades for face & eyes
face_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_eye.xml')

def getFace(frame):
	#Work on grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Find faces
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	#Filter largest face
	faceIndex = None
	maxW = -1
	for i,(x,y,w,h) in enumerate(faces):
		if w > maxW:
			faceIndex = i
			maxW = w

	if faceIndex != None:
		x,y,w,h = faces[faceIndex]
		#Draw rectangle around face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		faceImage = gray[y:y+h, x:x+w]
		return faceImage
	else:
		return None

#Assumes there is a face
def getEyes(face):
	
	#Find eyes in the face
	eyes = eye_cascade.detectMultiScale(
		face,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(20, 20),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	eyePics = []

	#Two max eyes
	for i,(ex,ey,ew,eh) in enumerate(eyes[:2]):
		#Focus on the center of the eye
		ew2 = int(ew*0.75)
		eh2 = int(eh*0.5)
		ex2 = int(ex+float(ew)/2-float(ew2)/2)
		ey2 = int(ey+float(eh)/2-float(eh2)/2)

		eye = face[ey2:ey2+eh2,ex2:ex2+ew2]
		
		#Draw rectangle around eyes
		#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		eyePics.append(eye)

	return eyePics

#Used to show/save NON-PROCESSED images in "raw"
def stream(record, showFrame):
	face_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_eye.xml')

	vs = WebcamVideoStream(src=0).start()
	
	while True:
		mouseX, mouseY = autopy.mouse.get_pos()
		
		frame = vs.read()
		frame = imutils.resize(frame, width=1200)
		face = getFace(frame)

		#If there is a face
		if face != None:

			#Find eyes
			eyes = getEyes(face)

			if record:
				for eye in eyes:
					save(eye, mouseX, mouseY, rawPath)
		
		#Show frame
		if showFrame:
			cv2.imshow('frame', imutils.resize(frame,width=300))

		#Threading ?
		waitMs = 50
		key = cv2.waitKey(waitMs) & 0xFF


#Main
#stream(record=False, showFrame=False)



