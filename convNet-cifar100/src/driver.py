import data_utils
import numpy as np
import convNet


def main(dataset):
	# Load the CIFAR-100 Data
	print "Loading data.."
	Xtr, Ytr1, Xte, Yte1 = data_utils.load_CIFAR100(dataset)
	print "Loaded data!"

	# No need to flatten the data for CNN
	Xtr = flatten(Xtr)
	Xte = flatten(Xte)

	mean_image = np.mean(Xtr, axis=0)

	Xtr = preProcess(Xtr, mean_image)
	Xte = preProcess(Xte, mean_image)

	# Make the labels of shape [numData, 20]
	Ntr = Ytr1.shape[0]
	Nte = Yte1.shape[0]
	Ytr = np.zeros([Ntr, 20])
	Yte = np.zeros([Nte, 20])
	Ytr[range(Ntr), Ytr1] += 1
	Yte[range(Nte), Yte1] += 1

	# Regroup the data into validation and training sets
	# 49k training, 1k validation
	assert Ntr == 50000
	Ntr = 49000
	convNet.runInstance(Xtr[:Ntr], Ytr[:Ntr], Xtr[Ntr:], Ytr[Ntr:], Xte, Yte)

""" Miscellaneous functions for preprocessing
"""
def preProcess(X, mean_image):
	""" Also divide by the maximum color value times 
	constant so that entries are in [-1, 1]
	"""
	X -= mean_image
	X /= 256 #*4
	return X

def flatten(X):
	# Bring every entry in the range 0-1
	N = X.shape[0]
	D = X.size / N
	X = X.reshape(N, D)
	return X

