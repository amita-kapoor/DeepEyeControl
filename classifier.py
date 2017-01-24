import threading
import time
from config import datasetImageSize, datasetMaxSerieLength, modelsPath, framesInHistory
from model import createModel
from datasetTools import padLSTM

def threaded(fn):
	def wrapper(*args, **kwargs):
		thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
		thread.start()
		return thread
	return wrapper

class Classifier:

	#What we want to reach
	predictionsPerSecond = 3

	#/!\ HARD-CODED
	recordingFPS = 25.

	#How many predictions to have average on motion
	averageWindowSize = int(predictionsPerSecond*framesInHistory/recordingFPS+1)

	print "Window size: at {} predicitions per seconds".format(predictionsPerSecond), averageWindowSize

	def __init__(self):
		self.X0 = None
		self.predictions = []
		self.lastPredictions = None
		self.model = createModel(nbClasses=3, imageSize=datasetImageSize, maxlength=datasetMaxSerieLength)
		print "Loading model parameters..."
		self.model.load(modelsPath+'eyeDNN_HD_SLIDE.tflearn')
		
		#History of last 3 predictions
		self.history = [None for i in range(averageWindowSize)]

	@threaded
	def startPredictions(self):
		while True:
			if self.X0 is not None:

				#Record start time
				t0 = time.time()

				#Read history
				X0 = list(self.X0)
				X1 = list(self.X1)

				#Pad inputs
				X0 = padLSTM(X0, maxlen=datasetMaxSerieLength, padding='post', value=0.)
				X1 = padLSTM(X0, maxlen=datasetMaxSerieLength, padding='post', value=0.)

				#Get predictions from the model with post padding			
				predictionSoftmax = self.model.predict([X0,X1])[0]
				predictedIndex = max(enumerate(predictionSoftmax), key=lambda x:x[1])[0]
				print "Prediction:", ["{0:.2f}".format(x) for x in predictionSoftmax], "->", predictedIndex
				
				self.predictions.append((predictionSoftmax,predictedIndex))
				self.lastPredictions = (predictionSoftmax,predictedIndex)

				#Crop beginning
				self.history = self.history[1:]
				self.history.append(predictionSoftmax)

				if None not in self.history:
					averageSoftmax = [sum(_)/float(len(self.history)) for _ in zip(*self.history)]
					averageIndex = max(enumerate(averageSoftmax), key=lambda x:x[1])[0]
					print "AVERAGE -------------------------:", ["{0:.2f}".format(x) for x in averageSoftmax], "->", averageIndex

				#FPS ?
				#Tiny sleep ?
				timeElapsed = time.time() - t0
				#Try to maintain 1./predictionsPerSecond s between each prediction
				time.sleep(max(0, 1./predictionsPerSecond - timeElapsed))





