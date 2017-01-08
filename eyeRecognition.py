import os
import cv2
import random
import string
import math
from random import shuffle
import numpy as np
import pickle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import datetime
from subprocess import PIPE, Popen
from trainModel import createModel
from config import datasetImageSize, cascadePath, modelsPath
from datasetTools import process
from imutils.video import WebcamVideoStream
import cv2
import imutils
from record import getFace, getEyes

def volumeUp():
	print "Volume up!"
	volume = getVolume()
	setVolume(volume+10)

def volumeDown():
	print "Volume down!"
	volume = getVolume()
	setVolume(volume+10)

def printPattern():
	print "Pattern!"

def getVolume():
	process = Popen("/usr/local/bin/vol info", stdout=PIPE, shell=True)
	out,error = process.communicate()
	volume = out.split(',')[0].split(':')[1]
	return int(volume)

def setVolume(value):
	process = Popen(["/usr/local/bin/vol out {}".format(value)], stdout=PIPE, shell=True)
	out,error = process.communicate()

#Used to show/save NON-PROCESSED images in "raw"
def main():

	#TODO REPLACE 5
	model = createModel(4,datasetImageSize)
	model.load(modelsPath+'eyeDNN5.tflearn')
	print "Model loaded"

	history = [-1,-1,-1,-1]

	face_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_eye.xml')

	vs = WebcamVideoStream(src=0).start()
	
	while True:
		
		frame = vs.read()
		frame = imutils.resize(frame, width=1200)
		face = getFace(frame)

		#If there is a face
		if face != None:
			#Find eyes
			eyes = getEyes(face)
			if len(eyes) == 2:
				predictedIndexes = []
				for eye in eyes:
					eye = process(eye)
					predictionSoftmax = model.predict(np.array(eye).reshape([-1,datasetImageSize,datasetImageSize,1]))[0]
					predictedIndex = max(enumerate(predictionSoftmax), key=lambda x:x[1])[0]
					if predictionSoftmax[predictedIndex] > 0.75:
						predictedIndexes.append(predictedIndex)

				#Two significant results
				if len(predictedIndexes) == 2 and predictedIndexes[0] == predictedIndexes[1]:
					print predictedIndexes[0], history
					#Don't add duplicates 110011 => 101
					if predictedIndexes[0] != history[-1]:
						history = history[1:]
						history.append(predictedIndexes[0])

				historyTxt = "".join(map(str,history))
				patterns = {"2020":volumeUp, "3131":volumeDown, "0312":printPattern}
				
				for pattern in patterns:
					if pattern in historyTxt:
						history = [-1 for i in range(len(history))]
						patterns[pattern]()
				

		#Threading ?
		waitMs = 50
		key = cv2.waitKey(waitMs) & 0xFF




main()








