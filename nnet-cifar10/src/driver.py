import data_utils
import network
import numpy as np
import matplotlib.pyplot as plt


def main(dataset):
	# Load the CIFAR Data
	print "Loading data.."
	Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10(dataset)
	print "Loaded data!"
	Xtr = flatten(Xtr)
	Xte = flatten(Xte)

	mean_image = np.mean(Xtr, axis=0)
	Xtr = preProcess(Xtr, mean_image)
	Xte = preProcess(Xte, mean_image)

	N, D = Xtr.shape
	vSize = N * 20 / 100  # Set aside 20% of data for validation

	# Create network and run training
	nn = network.TwoLayerNet(3072, 1024, 10)
	stats = nn.train(Xtr[vSize:], Ytr[vSize:], Xtr[:vSize], Ytr[:vSize], verbose=False)
	
	# Do not print stats..
	#print stats['train_acc_history']
	#print stats['loss_history']
	#print stats['val_acc_history']
	#plt.plot(stats['train_acc_history'])
	#plt.show()
	
	# Test accuracy
	print "Training accuracy: %.2f" % stats['train_acc_history'][-1]
	print "Validation accuracy: %.2f" % stats['val_acc_history'][-1]
	print "Testing accuracy: %.2f" % (nn.accuracy(Xte, Yte)*100)


""" Miscellaneous functions for preprocessing
"""
def PCAWhiteningCore(X):
	print X.shape
	cov = np.dot(X.T, X) / X.shape[0]
	U,S,V = np.linalg.svd(cov)
	Xrot = np.dot(X, U)
	#Xrot_reduced = np.dot(X, U[:,:100])
	Xwhite = Xrot / np.sqrt(S + 1e-5)
	print Xwhite.shape
	return Xwhite


def PCAWhitening(X):
	# This core operation on the entire matrix will take super long, so 
	# break it in managable chunks and work on it
	chunkSize = 100
	XW = np.zeros_like(X)
	for i in range(0, len(X), chunkSize) :
		print 'PCA Whitening chunk: %d' % i
		XW[i : i + chunkSize] += PCAWhiteningCore(X[i : i + chunkSize])
	return XW


def preProcess(X, mean_image):
	""" Also divide by the maximum color value times 
	constant so that entries are in [-1, 1]
	"""
	X -= mean_image
	#X = PCAWhitening(X) (Takes forever, dont do it!)
	X /= 256*4
	return X

def flatten(X):
	# Bring every entry in the range 0-1
	N = X.shape[0]
	D = X.size / N
	X = X.reshape(N, D)
	return X

