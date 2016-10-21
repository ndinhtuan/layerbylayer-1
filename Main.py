from Network2.NetworkFully import NetworkFully
from python_mnist.mnist import MNIST 
import numpy as np

#print(NetworkFully.__doc__)

net = NetworkFully()
net.addLayer(784, "None").addLayer(400, "ReLU").addLayer(400, "ReLU").addLayer(10, "sigmoid")
net.createLayers()
net.checkShapeWeights()

data = MNIST("python_mnist\data")
t = data.load_training()

net.train(t, 3, 50, 0.01, 0) 

#draw loss function
net.drawLoss()
net.saveWeight()

#Test and check
test = data.load_testing();
Xtest = np.array(test[0]);
Xtest = Xtest.T 
ytest = np.array([test[1]])
net.evalute(Xtest, ytest)