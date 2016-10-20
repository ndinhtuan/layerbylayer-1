class Layer:
	'''
		Class layer :
		input : input of layer, which is actiovation of previous layer
		ouput : ouput of layer, which goes through activation
		z : value of layer before going through activation
		nameOfAct : name of layer activation
		gradWeight : derivative of weight
		gradBias : derivate of bias 
		weight : weight of previous layer connect to this layer
		bias : bias of previous layer connect to this layer
	'''

	# init attributes of layer
	def __init__(self):
		self.ouput = None # activation, after going through activation
		self.z = None # backprop
		self.input = None #setInput
		self.nameOfAct = None #init
		self.weight = None #init
		self.bias = None #init
		self.gradWeight = None #backprop
		self.gradBias = None #backprop

	def setInput(self, input) :
		self.input = input

	# function forward compute ouput and z of this layer
	# compute z and ouput of layer, base on weight and bias 
	# return ouput of this layer
	def forward(self):
		pass

	#function backprop compute delta of previous layer and derivative of weight and bias, which is saved in gradWeight, gradBias
	#return delta previous layer.
	def backprop(self):
		pass	

	# update weight and bias
	def update(self):
		pass
