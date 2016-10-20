import numpy as np 
from .LayerFully import LayerFully
import time

class NetworkFully:
	'''
		class NetworkFully  include layers and fully connected
	'''
	def __init__(self):
		self.neuOfLayers = []
		self.nameOfActs = []
		self.layers = []

	def addLayer(self, numOfNeurons, nameOfAct):
		self.neuOfLayers.append(numOfNeurons);

		if len(self.nameOfActs) == 0:
			self.nameOfActs.append("None");
		else :
			self.nameOfActs.append(nameOfAct)

		return self

	def createLayers(self) :

		for i in range(len(self.nameOfActs) - 1):
			layer = LayerFully(self.nameOfActs[i + 1], self.neuOfLayers[i], self.neuOfLayers[i + 1])
			self.layers.append(layer)

	# Function feedforward compute result of model in final layer.
	# @ param X is input 
	# return result in final layer
	def feedforward(self, X):
		activation = X
		for layer in self.layers :
			layer.setInput(activation)
			activation = layer.forward()

		return activation

	#function runGD run stochastic gradient decent 
	# function will update all weight in each layer
	# return cost function in current processing.
	def runGD(self, X, y, alpha, decay):
		sizeOutput = self.neuOfLayers[-1]
		m = X.shape[1]
		h = self.feedforward(X)
		matY = np.zeros((sizeOutput, m))
		for i in range(m):
			matY[y[0][i]][i] = 1

		cost = 0
		cossTerm = (-1.0 / m) * sum(sum( matY*np.log2(h) + (1 - matY) * np.log2(1 - h) ))
		decayTerm = 0

		for layer in self.layers:
			decayTerm += decay * layer.getDecayTerm();

		cost = cossTerm + decayTerm 
		#print("M = {}. Term : {} and {}.\n".format(m, cossTerm, decayTerm))
		#print(self.layers[-1].getWeight())
		#time.sleep(5)

		delta = h - matY
		for i in range(len(self.layers)):
			delta = self.layers[-i-1].backprop(delta);
			self.layers[-i-1].update(alpha, decay);

		return cost

	def updateMiniBatch(self, miniBatch, alpha, decay):
		X = np.array(miniBatch[0])
		X = X.T
		y = np.array(miniBatch[1])
		#print("Shape: {} and {}.\n".format(X.shape, y.shape))

		cost = self.runGD(X, y, alpha, decay)
		print("Cost = {}.\n".format(cost))

	def train(self, dataTraining, epochs, batchSize, alpha, decay):
		X = np.array(dataTraining[0])
		X = (1 / 255.0) * X # image in range[0, 1)
		y = np.array([dataTraining[1]])

		m = X.shape[0]
		for epoch in range(epochs):
			print("Epoch {}:\n".format(epoch))
			miniBatches = [[X[k:k+batchSize, :], y[:, k:k+batchSize]] for k in range(0, m, batchSize)]

			for miniBatch in miniBatches:
				self.updateMiniBatch(miniBatch, alpha, decay)

	def checkShapeWeights(self):
		for layer in self.layers:
			print("{}.\n".format(layer.getShapeWeight()))

		