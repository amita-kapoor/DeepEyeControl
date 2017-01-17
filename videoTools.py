import cv2
from datasetTools import getDifferenceFrame
import numpy as np
from config import datasetImageSize

def getBlankFrameDiff():
	return np.zeros((datasetImageSize,datasetImageSize)).reshape([datasetImageSize,datasetImageSize,1])

def showDifference(t1,t0):
	#Resize for consistency
	t0 = cv2.resize(t0,(400,400))
	t1 = cv2.resize(t1,(400,400))

	#Diff between 0 and 255
	diff = getDifferenceFrame(t1,t0)
	diff = (diff+255.)/2.
	diff = diff.astype(t1.dtype)

	#Resize to show better
	t0 = cv2.resize(t0,(100,100))
	t1 = cv2.resize(t1,(100,100))
	diff = cv2.resize(diff,(100,100))

	#Display
	img = np.hstack((t0,t1,diff))

	t0 = cv2.equalizeHist(t0)
	t1 = cv2.equalizeHist(t1)

	#Diff between 0 and 255 to display
	diff = getDifferenceFrame(t1,t0)
	diff = (diff+255.)/2.
	diff = diff.astype(t1.dtype)

	#Resize to show better
	t0 = cv2.resize(t0,(100,100))
	t1 = cv2.resize(t1,(100,100))
	diff = cv2.resize(diff,(100,100))

	#Display
	processedImg = np.hstack((t0,t1,diff))

	cv2.imshow("Eyes", np.vstack((img,processedImg)))


def getFace(face_cascade, frame):
	#Find faces
	faces = face_cascade.detectMultiScale(
		frame,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(50, 50),
		maxSize=(110, 110),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	if len(faces) != 1:
		return None
	
	x,y,w,h = faces[0]

	return frame[y:y+h, x:x+w]

#Returns left eye then right (on picture)
def getEyes(eye_cascade, face):
	
	#Find eyes in the face
	eyes = eye_cascade.detectMultiScale(
		face,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(20, 20),
		maxSize=(40, 40),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	if len(eyes) != 2:
		return None

	leftEye = None
	leftEyeX = 999
	rightEye = None

	#Two eyes
	for ex,ey,ew,eh in eyes:
		#Focus on the center of the eye
		ew2 = int(ew*0.75)
		eh2 = int(eh*0.5)
		ex2 = int(ex+float(ew)/2-float(ew2)/2)
		ey2 = int(ey+float(eh)/2-float(eh2)/2)

		eye = face[ey2:ey2+eh2,ex2:ex2+ew2]
		
		#New left eye
		if ex < leftEyeX:
			if leftEye is not None:
				rightEye = leftEye
				leftEye = eye
			else:
				leftEye = eye
				leftEyeX = ex
		else:
			if leftEye is not None:
				rightEye = eye

	return leftEye, rightEye










