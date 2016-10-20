import numpy as np 
import math

def initWeight(layerIn, layerOut):
	epsilon = math.sqrt(6 / (layerIn + layerOut))
	result = 2 * epsilon * np.random.rand(layerOut, layerIn) - epsilon

	return result

def sigmoid(z):
	arr = np.array(z)
	a = 1 / (1 + math.e**(-arr))
	return a

def ReLU(z):
	arr = np.array(z)
	a = arr * (arr > 0)

	return a

def sigmoidGrad(z):
	z = np.array(z)
	return sigmoid(z) * (1 - sigmoid(z))

def ReLUGrad(z):
	z = np.array(z)
	return 1 * (z > 0)