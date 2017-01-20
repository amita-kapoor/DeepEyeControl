import threading
import time
from config import datasetImageSize, datasetMaxSerieLength, modelsPath
from model import createModel
from datasetTools import padLSTM

def threaded(fn):
	def wrapper(*args, **kwargs):
		thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
		thread.start()
		return thread
	return wrapper

class Classifier:

	def __init__(self):
		self.X0 = None
		self.predictions = []
		self.lastPredictions = None
		self.model = createModel(nbClasses=3, imageSize=datasetImageSize, maxlength=datasetMaxSerieLength)
		print "Loading model parameters..."
		self.model.load(modelsPath+'eyeDNN_HD_SLIDE.tflearn')
		#History of last 3 predictions
		self.history = [None, None, None]

	@threaded
	def startPredictions(self):
		while True:
			if self.X0 is not None:
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
					averageSoftmax = [(self.history[0][i] + self.history[1][i] + self.history[2][i])/3. for i in range(3)]
					averageIndex = max(enumerate(averageSoftmax), key=lambda x:x[1])[0]
					print "AVERAGE -------------------------:", ["{0:.2f}".format(x) for x in averageSoftmax], "->", averageIndex

				#FPS ?
				#Tiny sleep ?
				time.sleep(0.2)





