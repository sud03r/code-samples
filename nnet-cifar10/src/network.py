import numpy as np
import random
import utils
import matplotlib.pyplot as plt
import scipy.misc as spm

class TwoLayerNet(object):

	# D (input), H (hidden layer), C (output). The activation function at H is ReLU, 
	# activation function at C is softmax.
	
	def __init__(self, input_size, hidden_size, output_size, std=1e-2, af='ReLU'):
		
		# Define the activation functions
		self.activation = {'sig' : utils.sigmoid, 'tanh' : utils.tanh, 
							'ReLU' : utils.ReLU, 'pReLU' : utils.pReLU }
		self.activationPrime = {'sig' : utils.sigmoid_prime, 'tanh' : utils.tanh_prime, 
							'ReLU' : utils.ReLU_prime, 'pReLU' : utils.pReLU_prime }
		
		self.params = {}
		self.params['W1'] = utils.initializeWeights(input_size, hidden_size)
		self.params['b1'] = utils.initializeBiases(hidden_size)
		self.params['W2'] = utils.initializeWeights(hidden_size, output_size)
		self.params['b2'] = utils.initializeBiases(output_size)
		self.params['af'] = af

	def loss(self, X, y=None, reg=0.0, dropout=False):
		# Unpack variables from the params dictionary
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		f = self.activation[self.params['af']]
		fp = self.activationPrime[self.params['af']]
		N, D = X.shape

		dropout = dropout and y is None # Only dropout for training data
		pActive = 1 # Probability of keeping the next active

		# Compute the scores in forward pass
		z1 = X.dot(W1) + b1	
		# Apply the activation function
		a1 = f(z1)
		
		if dropout :
			U1 = (np.random.rand(*a1.shape) < pActive) / pActive
			a1 *= U1

		z2 = a1.dot(W2) + b2
		# Apply Softmax
		exp_scores = np.exp(z2)
		eps = 1e-6 # To avoid divide by zero issues.
		scores = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + eps)

		# If the targets are not given then jump out, we're done
		if y is None:
			return scores

		# Compute average loss and compute Regularization
		assert N == y.size
		loss = -np.log(scores[range(N), y])
		loss = np.average(loss) + 0.5 * reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

		# Backward pass: compute gradients
		""" Recall that delta3 = y_op - y_actual. However for a given input x [3072x1]
			the output y is an integer 'pos' \in [0, 9] indicating its correct class. But y_op
			is a 10 x 1 vector of probabilities. So we can imagine y_actual to also be
			a 10 x 1 vector with all zeros but the value at 'pos' to be 1.
			Then, d3 [range(N), y] -= 1 achieves the desired behavior.
		"""
		d3 = scores
		d3 [range(N), y] -= 1
		dW2 = a1.T.dot(d3)
		db2 = np.sum(d3, axis=0)
		d2 = d3.dot(W2.T) * fp(z1)
		dW1 = np.dot(X.T, d2)
		db1 = np.sum(d2, axis=0)
		
		assert dW2.shape == W2.shape
		assert db2.shape == b2.shape
		assert dW1.shape == W1.shape
		assert db1.shape == b1.shape

		# Add regularization loss
		dW2 += reg * W2
		dW1 += reg * W1

		grads = { 'W1' : dW1, 'b1' : db1, 'W2' : dW2, 'b2' : db2 }
		return loss, grads


	def train(self, X, y, X_val, y_val,
		learning_rate= 5e-3, learning_rate_decay=0.998,
		reg=5e-3, num_iters=100,
		batch_size=1000, paramUpdates = 'Momentum',
		verbose=False):

		num_train = X.shape[0]
		iterations_per_epoch = max(num_train / batch_size, 1)
		num_iters = 30 * iterations_per_epoch

		# Use SGD to optimize the parameters in self.model
		loss_history = []
		train_acc_history = []
		val_acc_history = []

		""" We will try three different ways of parameter updates
			1. Vanilla Update : p -=  learning_rate * grads[p]
			2. RMSprop, uses the cache and learning_rate_decay
			3. Momentum update
		"""

		# Temporary variables for RMSProp and Momentum
		mu = 0.5
		eps = 1e-6
		cache = {}
		for p in ('W1', 'b1', 'W2', 'b2'):
			cache[p] = np.zeros_like(self.params[p])

		for it in xrange(num_iters):
			# Create a random minibatch of training data and labels
			randomIndices = random.sample(range(num_train), batch_size)
			X_batch = X[randomIndices]
			y_batch = y[randomIndices]

			# Compute loss and gradients using the current minibatch
			loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
			loss_history.append(round(loss, 3))


			# Use the gradients in the grads dictionary to update the
			# parameters of the network (stored in the dictionary self.params)
			
			for p in ('W1', 'b1', 'W2', 'b2'):
				change = learning_rate * grads[p]
				if paramUpdates is 'RMSProp':
					cache[p] = learning_rate_decay * cache[p] + (1 - learning_rate_decay) * (grads[p]**2)
					self.params[p] -= change / (np.sqrt(cache[p]) + eps)
				elif paramUpdates is 'Momentum':
					cache[p] = mu * cache[p] - change # v = cache[p]
					self.params[p] += cache[p]
				else:
					self.params[p] -= change

			if verbose and it % 5 == 0:
				print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
			# Every epoch, check train and val accuracy and decay learning rate.
			
			if it % 50 == 0:
				# Check accuracy
				train_acc = round((self.predict(X_batch) == y_batch).mean() * 100, 3)
				val_acc = round((self.predict(X_val) == y_val).mean() * 100, 3)
				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)
				print 'train_acc %.2f val_acc %.2f' % (train_acc, val_acc)

			learning_rate *= learning_rate_decay

		return {
			'loss_history': loss_history,
			'train_acc_history': train_acc_history,
			'val_acc_history': val_acc_history,
			}

	def predict(self, X):
		probs = self.loss(X)
		y_pred = np.argmax(probs, axis=1)
		return y_pred

	def accuracy(self,X,y):
		acc = (self.predict(X) == y).mean()
		return acc

