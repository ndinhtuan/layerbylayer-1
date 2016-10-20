from Network2.NetworkFully import NetworkFully
from python_mnist.mnist import MNIST 

net = NetworkFully()
net.addLayer(784, "None").addLayer(400, "ReLU").addLayer(400, "ReLU").addLayer(10, "sigmoid")
net.createLayers()
net.checkShapeWeights()

data = MNIST("python_mnist\data")
t = data.load_training()

net.train(t, 3, 50, 0.01, 5e-4)        