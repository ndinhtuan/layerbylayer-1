import numpy as np 
from .LayerFully import LayerFully
import time
import matplotlib.pyplot as plt
import copy
from .NetLib import initWeight, sigmoid, ReLU, sigmoidGrad, ReLUGrad

class NetworkFully:
	'''
		class NetworkFully  include layers and fully connected
	'''
	def __init__(self):
		self.neuOfLayers = []
		self.nameOfActs = []
		self.layers = []
		self.lossToDraw = []

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

		if self.nameOfActs[-1] == "sigmoid":
			prime = sigmoidGrad(self.layers[-1].z)
		elif self.nameOfActs[-1] == "ReLU":
			prime = ReLUGrad(self.layers[-1].z)
		else : 
			print("Cannot find activation \n")
		delta = (h - matY) * prime

		for i in range(len(self.layers)):
			delta = self.layers[-i-1].backprop(delta)
			self.layers[-i-1].update(alpha, decay)

		return cost

	def updateMiniBatch(self, miniBatch, alpha, decay):
		X = np.array(miniBatch[0])
		X = X.T
		y = np.array(miniBatch[1])

		cost = self.runGD(X, y, alpha, decay)
		self.lossToDraw.append(cost)
		print("Cost = {}.\n".format(cost))

	def train(self, dataTraining, epochs, batchSize, alpha, decay):
		X = np.array(dataTraining[0])
		X = (1 / 255.0) * X # image in range[0, 1)
		y = np.array([dataTraining[1]])

		m = X.shape[0]
		checkedGrad = False
		for epoch in range(epochs):
			print("Epoch {}:\n".format(epoch))
			miniBatches = [[X[k:k+batchSize, :], y[:, k:k+batchSize]] for k in range(0, m, batchSize)]

			for miniBatch in miniBatches:
				if checkedGrad == False:
					print("Checking gradient : \n")
					checkedGrad = True
					tmpX = np.array(miniBatch[0])
					tmpX = tmpX.T 
					tmpY = np.array(miniBatch[1]) 
					self.checkGradientDescent(tmpX, tmpY)
					input('Press somthing to continue ...\n')

				self.updateMiniBatch(miniBatch, alpha, decay)

	def checkShapeWeights(self):
		for layer in self.layers:
			print("{}.\n".format(layer.getShapeWeight()))

	#Cost function don't update
	def costNNs(self, X, y):
		outputSize = self.neuOfLayers[-1]
		m = X.shape[1]
		matY = np.zeros((outputSize, m))

		h = self.feedforward(X)

		if self.nameOfActs[-1] == "sigmoid":
			prime = sigmoidGrad(self.layers[-1].z)
		elif self.nameOfActs[-1] == "ReLU":
			prime = ReLUGrad(self.layers[-1].z)
		else : 
			print("Cannot find activation \n")
		delta = (h - matY) * prime

		for i in range(len(self.layers)):
			delta = self.layers[-i-1].backprop(delta)

		cost = (-1.0 / m) * sum(sum( matY*np.log2(h) + (1 - matY)*np.log2(1 - h) ))
		return cost

	# Check deriviate of coss on Weight1
	def checkGradientDescent(self, X, y):
		self.costNNs(X, y)
		normGrad = self.layers[2].getGrad()

		#compute numeric grad
		epsilon = 1e-4
		numericGrad = []

		for i in range(5):
			self.layers[2].getWeight()[0][i] += epsilon
			cost1 = self.costNNs(X, y)
			self.layers[2].getWeight()[0][i] -= 2*epsilon
			cost2 = self.costNNs(X, y)
			numericGrad.append((cost1 - cost2) / (2 * epsilon))
			self.layers[2].getWeight()[0][i] += epsilon

		for i in range(5):
			print("{} - {}.\n".format(normGrad[0][i], numericGrad[i]))



		

	def predict(self, X):
		h = self.feedforward(X);

		return h.argmax(axis=0)

	def evalute(self, Xtest, ytest):
		pre = self.predict(Xtest)
		print("Accurate : {}.\n".format(np.mean(pre == ytest)))


	def drawLoss(self):
		plt.plot(self.lossToDraw);
		plt.ylabel("Loss function")
		plt.xlabel("#.Iterations")
		plt.show()

	def saveWeight(self):
		listWeight = []
		tmp = []
		for layer in self.layers :
			tmp.append(layer.weight)
			tmp.append(layer.bias)

			listWeight.append(tmp)
		np.save("weight", listWeight)
