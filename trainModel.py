import pickle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import datetime
from config import modelsPath, datasetPath, datasetImageSize

def createModel(nbClasses,imageSize):
	print("[+] Creating model...")
	convnet = input_data(shape=[None, imageSize, imageSize, 1], name='input')
	convnet = fully_connected(convnet, 512, activation='elu')
	convnet = dropout(convnet, 0.5)
	convnet = fully_connected(convnet, 256, activation='elu')
	convnet = dropout(convnet, 0.5)
	convnet = fully_connected(convnet, nbClasses, activation='softmax')
	convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0005)
	model = tflearn.DNN(convnet)
	return model


def fitModel():
	#Load dataset
	X = pickle.load(open(datasetPath+"X4_4_2_{}.p".format(datasetImageSize), "rb" ))
	y = pickle.load(open(datasetPath+"y4_4_2_{}.p".format(datasetImageSize), "rb" ))

	print len(X), len(y)
	model = createModel(nbClasses=4, imageSize=datasetImageSize)
	runId = "Eye Tracking - " + str(datetime.datetime.now().time())
	model.fit(X, y, n_epoch=300, batch_size=128, shuffle=True, validation_set=0.3, snapshot_step=5000, show_metric=True, run_id=runId)

	#Save
	model.save(modelsPath+"eyeDNN5.tflearn")

#fitModel()


