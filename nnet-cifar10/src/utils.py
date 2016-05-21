import numpy as np

#### Activation functions
""" Define all the activation functions and their deriviatives.
	We will use them later in our network
""" 

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
	return np.tanh(z)

def tanh_prime(z):
	a = np.tanh(z)
	return (1 - a*a)

def ReLU(z):
	return np.maximum(0, z)

def ReLU_prime(z):
	return z > 0

""" Pick the parameter to be alpha = 0.5
"""

def pReLU(z):
	a = 0.75
	return np.maximum(0, z * a)

def pReLU_prime(z):
	a = 0.75
	return (z > 0) * a


""" The following functions are different ways of
	initializing the weights and biases
"""
def initializeWeights(nIn, nOut, std=1e-2, mode='normalize'):
	W_i = std * np.random.randn(nIn, nOut)
	if mode is 'normalize':
		W_i *= np.sqrt(2.0 / nIn)
	return W_i

def initializeBiases(nOut, std=1e-2):
	return np.ones(nOut) * std
