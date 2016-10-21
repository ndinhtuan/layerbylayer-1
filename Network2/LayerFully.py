from .Layer import Layer
from .NetLib import initWeight, sigmoid, ReLU, sigmoidGrad, ReLUGrad
import numpy as np 

class LayerFully(Layer):
	'''
	class LayerFully compute on fully connected between layers : Neural Network fully
	'''

	# __init__ init weight and bias of layer , and set nameOfAct
	# @ parameter nameOfAct : name of activation apply for this layer (sigmoid, ReLU, ...)
	# @ Lin                 : number of neurons in input layer
	# @ Lout				: number of neurons in this layer
	def __init__(self, nameOfAct, Lin, Lout):
		#if self.weight is None:
		self.weight = initWeight(Lin, Lout) 
		self.bias = np.zeros((Lout, 1))

		self.nameOfAct = nameOfAct

	def setInput(self, input):
		self.input = input

	#Override
	def forward(self):
		self.z = np.dot(self.weight, self.input) + self.bias

		if self.nameOfAct == "sigmoid":
			self.output = sigmoid(self.z)
		elif self.nameOfAct == "ReLU":
			self.output = ReLU(self.z)
		else:
			print("Cannot find activation " + self.nameOfAct)
			return None

		return self.output

	#Overload
	#backpop compute previous delta and return it, on beside  it also compute derivative of bias and weight
	#@parameter delta is matrix of this derivative Coss function on this y
	def backprop(self, delta):
		if self.nameOfAct == "sigmoid" :
			prime = sigmoidGrad(self.input)
		elif self.nameOfAct == "ReLU" :
			prime = ReLUGrad(self.input)
		else:
			print("Cannot find activation " + self.nameOfAct)
			return None

		preDelta = np.dot(self.weight.T, delta) * prime

		self.gradWeight = np.dot(delta, self.input.T)
		self.gradBias = np.array([(sum(delta.T))]).T

		return preDelta

	def update(self, alpha, decay):
		self.gradWeight += decay * self.gradWeight

		self.weight -= alpha * self.gradWeight
		self.bias -= alpha * self.gradBias

	#function get sum of quaratic weight
	def getDecayTerm(self):
		return (1 / 2.0) * sum(sum(self.weight ** 2))

	def getShapeWeight(self):
		return self.weight.shape 

	def getWeight(self):
		return self.weight

	def changeWeight(self, row, col, epsilon):
		self.weight[row][col] += epsilon

	def getGrad(self):
		return self.gradWeight

