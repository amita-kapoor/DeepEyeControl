import os
import cv2
import random
import string
import math
from random import shuffle
import numpy as np
import pickle
from config import rawPath, datasetPath, screenSize, datasetImageSize, processedPath, gridSize

#All processing on the image
def process(eye):
	eye =  cv2.resize(eye,(datasetImageSize, datasetImageSize), interpolation = cv2.INTER_CUBIC)
	eye = normalize(eye)
	return eye

#Corrects lightning
def normalize(eye):
	hist,bins = np.histogram(eye.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	eye = cdf[eye]
	return eye

#Flip
def flip(eye, x ,y):
	flippedEyes = []
	flippedEyes.append((eye,x,y))
	flippedEyes.append((cv2.flip(eye,1),screenSize[0] - x,y))
	return flippedEyes

#Rotate image and zoop
def rotate(eye, x, y):
	(h, w) = eye.shape[:2]
	center = (w / 2, h / 2)
	rotated = []
	for angle,zoom in [(-5,1.1),(5,1.1)]:
		# rotate the image by 180 degrees
		M = cv2.getRotationMatrix2D(center, angle, zoom)
		rotatedEye = cv2.warpAffine(eye, M, (w, h))
		rotated.append((rotatedEye,x,y))
	return rotated

#Flip, rotate, etc.
def augment(eye,x,y):
	augmentedEyes = []
	
	#Original
	flippedEyes = flip(eye,x,y)

	#TODO ROTATION USEFUL ?
	#Rotate
	for flippedEye, flippedX, flippedY in flippedEyes:
		for rotatedEye, rotatedX, rotatedY in rotate(flippedEye,flippedX,flippedY):
			augmentedEyes.append((rotatedEye, rotatedX, rotatedY))
	return augmentedEyes

#Writes image at desired path
def save(eye,x,y,destinationPath):
	randomHash = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
	name = "{}_{}_{}.png".format(x,y,randomHash)
	cv2.imwrite(destinationPath+name,eye)

####### GENERATE PROCESSED & AUGMENTED DATASET FROM RAW #######

#Reads from RAW and writes to PROCESSED
def generateProcessedImages():
	files = os.listdir(rawPath)
	files = [file for file in files if file.endswith(".png")]
	print len(files), "files found."

	for filename in files:
		eye = cv2.imread(rawPath+filename,0)
		eyeX, eyeY = map(int,filename[:-4].split("_")[-2:])

		processedEye = process(eye)
		augmentedEyes = augment(processedEye,eyeX,eyeY)
		for augmentedEye, x, y in augmentedEyes:
			save(augmentedEye, x, y, processedPath)

####### GENERATE (X,y) DATASET #########

#Grid index from top left to bottom right
def getIndex(gridWidth,gridHeight,imgx,imgy):
	xgrid = math.floor(float(imgx)/(float(screenSize[0])/gridWidth))
	ygrid = math.floor(float(imgy)/(float(screenSize[1])/gridHeight))
	index = gridWidth*ygrid + xgrid
	return index

#Creates pickle dataset (X,y)
def generateDataset(gridWidth,gridHeight): 
	files = os.listdir(processedPath)
	files = [file for file in files if file.endswith(".png")]

	shuffle(files)

	print len(files), "files found."

	X, y = [], []
	for filename in files:
		img = cv2.imread(processedPath+filename,0)		
		imgx,imgy = map(int,filename[:-4].split("_")[:-1])

		index = getIndex(gridWidth,gridHeight,imgx,imgy)
	
		conv = {0.:0,3.:1,4.:2,7.:3}

		if index in conv:
			newIndex = conv[index]
		else:
			newIndex = 4

		if newIndex < 4:
			#Add to dataset
			X.append(np.array(img.reshape([datasetImageSize,datasetImageSize,1])))

			#One hot
			label = [1. if i == newIndex else 0. for i in range(4)]
			y.append(np.array(label))

	pickle.dump(X, open(datasetPath+"X4_{}_{}_{}.p".format(gridWidth,gridHeight,datasetImageSize), "wb" ))
	pickle.dump(y, open(datasetPath+"y4_{}_{}_{}.p".format(gridWidth,gridHeight,datasetImageSize), "wb" ))


#generateDataset(*gridSize)
#generateProcessedImages()




