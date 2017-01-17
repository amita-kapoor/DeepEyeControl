import os
import cv2
import random
import string
import math
from random import shuffle
import numpy as np
import pickle
import tflearn
from subprocess import PIPE, Popen
from model import createModel
from config import datasetImageSize, cascadePath, modelsPath
from datasetTools import process, padLSTM, getDifferenceFrame
from imutils.video import WebcamVideoStream
import cv2
import imutils
import time
from detector import Detector
from videoTools import showDifference, getBlankFrameDiff

def detectPattern(predictionSoftmax):
	predictedIndex = max(enumerate(predictionSoftmax), key=lambda x:x[1])[0]

	#Only keep confident predictions
	if predictionSoftmax[predictedIndex] > 0.45:

		#If change/not sure
		if predictedIndex != predictionsHistory[-1] and predictionsHistory[-1] != -1:
			#Reset
			print "-Reset-"
			predictionsHistory = [-1 for i in range(15)]

		print predictedIndex, predictionsHistory
		predictionsHistory = predictionsHistory[1:]
		predictionsHistory.append(predictedIndex)

		if predictionsHistory.count(0) > 8:
			print "ITS A GAMAA"
		elif predictionsHistory.count(1) > 8:
			print "ITS A ZZZZZ"
		elif predictionsHistory.count(2) > 8:
			print "ITS A _____"

	#Check if history matches a pattern
	historyTxt = "".join(map(str,history))
	patterns = {"2020":pattern1, "0321":pattern2, "0123":pattern3}
	for pattern in patterns:
		if pattern in historyTxt:
			history = [-1 for i in range(len(history))]
			patterns[pattern]()

#Display whole history as seen by LSTM
def displayHistoryDiffs(framesDiffHistory, fps):
	framesDiffHistoryImages = []
	for diff, _ in framesDiffHistory:
		if diff is None:
			diff = getBlankFrameDiff()
		else:
			diff = (diff.astype(float)+255.)/2.
		framesDiffHistoryImages.append(diff.astype(np.uint8))

	img = None
	for rowIndex in range(8):
		#Debug history
		rowImg = np.hstack(framesDiffHistoryImages[rowIndex*8:rowIndex*8+8])
		img = img = np.vstack((img,rowImg)) if img is not None else rowImg
	
	cv2.putText(img,"FPS: {}".format(int(fps)),(3,9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
	cv2.imshow("Frame history",img)

# Show current eyes with diff
def displayCurrentDiff(eye0, eye1, eye0previous, eye1previous, stopFrame=False):
	current = np.hstack((eye0.astype(np.uint8),eye1.astype(np.uint8)))
	last = np.hstack((eye0previous.astype(np.uint8),eye1previous.astype(np.uint8)))
	showDifference(current,last)
	
	# Debug frame by frame
	if stopFrame:
		cv2.waitKey(0)



def main(displayHistory=True):
	#History of size 30
	predictionsHistory = [-1 for i in range(15)]
	framesDiffHistory = [[None,None] for i in range(64)]
	lastEyes = None

	#Initializa webcam
	vs = WebcamVideoStream(src=0).start()
	
	t0 = -1
	frameID = 0

	detector = Detector()

	#Load model
	#model = createModel(nbClasses=3, imageSize=datasetImageSize, maxlength=100)
	#print "Loading model parameters..."
	#model.load(modelsPath+'eyeDNN_HD.tflearn')

	print "Starting eye recognition..."
	while True:

		dt =  time.time() - t0
		fps = 1/dt
		t0 = time.time()

		#Limit framerate
		waitMs = 5
		key = cv2.waitKey(waitMs) & 0xFF

		fullFrame = vs.read()
		fullFrame = cv2.cvtColor(fullFrame, cv2.COLOR_BGR2GRAY)
		frame = imutils.resize(fullFrame, width=300)

		faceBB = detector.getFace(frame)

		#If there is a face
		if faceBB is None:
			#Invalidate eyes bounding box as all will change (TODO REMOVE??)
			lastEyes = None
			detector.resetEyesBB()
			continue

		#Get small face coordinates
		x,y,w,h = faceBB
		face = frame[y:y+h, x:x+w]

		#Apply to fullscale
		xScale = fullFrame.shape[1]/frame.shape[1]
		yScale = fullFrame.shape[0]/frame.shape[0]
		x,y,w,h = x*xScale,y*yScale,w*xScale,h*yScale
		fullFace = fullFrame[y:y+h, x:x+w]

		#Find eyes
		eyes = detector.getEyes(fullFace)

		if eyes is None:
			#Reset last eyes
			lastEyes = None
			continue

		eye0, eye1 = eyes

		#Process (histograms, size)			
		eye0 = process(eye0)
		eye1 = process(eye1)
		
		#Reshape
		eye0 = np.reshape(eye0,[datasetImageSize,datasetImageSize,1])
		eye1 = np.reshape(eye1,[datasetImageSize,datasetImageSize,1])

		#We have a recent picture of the eyes
		if lastEyes is not None:
			#Load previous eyes
			eye0previous, eye1previous = lastEyes

			#Compute diffs
			diff0 = getDifferenceFrame(eye0, eye0previous)
			diff1 = getDifferenceFrame(eye1, eye1previous)

			displayDiff = False
			if displayDiff:
				displayCurrentDiff(eye0,eye1,eye0previous,eye1previous,stopFrame=False)

			#Crop beginning then add new
			framesDiffHistory = framesDiffHistory[1:]
			framesDiffHistory.append([diff0,diff1])

		#Keep current as last frame
		lastEyes = [eye0, eye1]

		#Not time consuming
		if displayHistory:
			displayHistoryDiffs(framesDiffHistory, fps)

		if [None,None] in framesDiffHistory:
			print "History filling..."
		else:
			#Extract each eyes
			X0, X1 = zip(*framesDiffHistory)

			#Reshape as a tensor (NbExamples,SerieLength,Width,Height,Channels)
			X0 = np.reshape(X0,[-1,len(framesDiffHistory),datasetImageSize,datasetImageSize,1])
			X1 = np.reshape(X1,[-1,len(framesDiffHistory),datasetImageSize,datasetImageSize,1])

			#Make shape (NbExamples,MaxLength,Width,Height,Channels)
			X0_post = padLSTM(X0, maxlen=100, padding='post', value=0.)
			X1_post = padLSTM(X1, maxlen=100, padding='post', value=0.)

			#X0_pre = padLSTM(X0, maxlen=100, padding='pre', value=0.)
			#X1_pre = padLSTM(X1, maxlen=100, padding='pre', value=0.)

			#print "------------------------------------"

			#Get predictions from the model with pre/post padding
			#predictionSoftmax_pre = model.predict([X0_pre,X1_pre])[0]
			#predictedIndex_pre = max(enumerate(predictionSoftmax_pre), key=lambda x:x[1])[0]
			
			#predictionSoftmax_post = model.predict([X0_post,X1_post])[0]
			#predictedIndex_post = max(enumerate(predictionSoftmax_post), key=lambda x:x[1])[0]
			
			#print "Pre: ", ["{0:.2f}".format(x) for x in predictionSoftmax_pre], "->", predictedIndex_pre
			#print "Post:", ["{0:.2f}".format(x) for x in predictionSoftmax_post], "->", predictedIndex_post
			
			##Average predictions
			#predictionSoftmax_avg = [predictionSoftmax_pre[i]*0.5 + predictionSoftmax_post[i]*0.5 for i in range(len(predictionSoftmax_pre))]
			#predictedIndex_avg = max(enumerate(predictionSoftmax_avg), key=lambda x:x[1])[0]
			#print "Avg: ", ["{0:.2f}".format(x) for x in predictionSoftmax_avg], "->", predictedIndex_avg

			#Handle verified patterns from history
			#detectPattern(predictionSoftmax)	




if __name__ == "__main__":
	main(displayHistory=True)








