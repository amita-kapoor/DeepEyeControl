import os
import cv2
import cv2
from config import rawPath, processedPath, screenSize, datasetPath, datasetImageSize
import matplotlib.pyplot as plt
import pickle

def showRawDatasetDistribution():
	files = os.listdir(rawPath)
	files = [file for file in files if file.endswith(".png")]

	fig, ax = plt.subplots()
	ax.grid(False)

	print len(files), "files in raw dataset."
	for filename in files:
		x, y = map(int,filename[:-4].split('_')[-2:])
		#Y is towards bottom
		y = screenSize[1] - y
		ax.scatter(x, y, color='r', alpha=0.3)

	plt.show()

def showClassDistribution():
	y = pickle.load(open(datasetPath+"yo_4_2_{}.p".format(datasetImageSize), "rb" ))
	counter = {0:0,1:0,2:0,3:0,4:0}
	for sample in y:
		sampleClass = max(enumerate(sample), key=lambda x:x[1])[0]
		counter[sampleClass] = counter[sampleClass] + 1
	print counter

def showMeanImage(): 
	files = os.listdir(processedPath)
	files = [file for file in files if file.endswith(".png")]

	meanImg = None
	nbFiles = len(files)
	ratio = 1./(nbFiles*255)

	for filename in files:
		img = cv2.imread(processedPath+filename,0)
		if meanImg == None:
			meanImg = img*ratio
		else:
			meanImg = cv2.add(meanImg,img*ratio)

	cv2.imshow("mean",meanImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#showRawDatasetDistribution()
showClassDistribution()















