import sys
import driver
import tensorflow as tf

###############################################################
#### Helper functions for our Convolutional Neural network ####
###############################################################


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, size=2):
	return tf.nn.max_pool(x, ksize=[1, size, size, 1],strides=[1, size, size, 1], padding='SAME')

###############################################################
### Construct and run the network #############################
###############################################################

def runInstance(Xtr, Ytr, Xv, Yv, Xte, Yte) :
		
	""" Parameters to our Convolutional Network """

	LABELS = 20 # Finally we have 20 coarse labels
	convSize = [8, 16] # Size of our convolutional layers
	fSize = 5 # Filter Size
	fcSize = 512 # Size of the fully connected layer
	finalSz = 8 # After two layers of pooling with padding = SAME, 32x32 image will have final size 8 x 8
	numIterations = 1000 # convnets take long time to train!
	learning_rate_decay = 0.999 # Decay the learning rate every epoch
	reg = 1e-5 # We get better (almost similar) results without Regularization.


	sess = tf.InteractiveSession()

	""" Make learning_rate a tensor so that we can update it every epoch"""
	learning_rate = tf.placeholder(tf.float32, shape=[])

	images = tf.placeholder(tf.float32, shape=[None, 3072])
	labels = tf.placeholder(tf.float32, shape=[None, 20])
	x_image = tf.reshape(images, [-1, 32, 32, 3])


	""" The Convolution + Pooling layer 1 """
	c1 = convSize[0]
	W_c1 = weight_variable([fSize, fSize, 3, c1])
	b_c1 = bias_variable([c1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_c1) + b_c1)
	h_pool1 = max_pool(h_conv1)

	""" The Convolution + Pooling layer 2 """
	c2 = convSize[1]
	W_c2 = weight_variable([fSize, fSize, c1, c2])
	b_c2 = bias_variable([c2])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_c2) + b_c2)
	h_pool2 = max_pool(h_conv2)

	""" The fully Connected layer + softmax """

	numInputs = finalSz * finalSz * c2
	W_fc = weight_variable([numInputs, fcSize])
	b_fc = bias_variable([fcSize])

	h_pool2_fcIN = tf.reshape(h_pool2, [-1, numInputs])  # Adjust size
	h_fc = tf.nn.relu(tf.matmul(h_pool2_fcIN, W_fc) + b_fc)

	""" Dropout layer """
	keep_prob = tf.placeholder(tf.float32)
	h_drop = tf.nn.dropout(h_fc, keep_prob)

	""" Softmax layer """
	W_sm = weight_variable([fcSize, 20])
	b_sm = bias_variable([20])
	y_conv = tf.nn.softmax(tf.matmul(h_drop, W_sm) + b_sm) # Final output

	# Note that labels is [N, 20] size vector with y[i] = 1 if i is the correct label, else 0
	softmaxLoss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y_conv), reduction_indices=[1]))

	# Add regularization loss
	softmaxLoss = softmaxLoss + reg * (tf.nn.l2_loss(W_sm) + tf.nn.l2_loss(W_fc) 
									  + tf.nn.l2_loss(W_c2) + tf.nn.l2_loss(W_c1))

	# Train using GD
	#optMethod = tf.train.AdamOptimizer(learning_rate)
	optMethod = tf.train.GradientDescentOptimizer(learning_rate)
	train_step = optMethod.minimize(softmaxLoss)

	# Measure accuracy
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess.run(tf.initialize_all_variables())

	batch_size = 1000
	num_train = Xtr.shape[0]
	iterations_per_epoch = max(num_train / batch_size, 1)
	learnRate = 0.1 # Initial learning rate

	lossHistory = []
	accHistory = []

	for i in xrange(numIterations):
		
		# Decay the learning rate
		if i % iterations_per_epoch == 0:
			 learnRate *= learning_rate_decay

		offset = (i * batch_size) % (Ytr.shape[0] - batch_size)
		batch_data = Xtr[offset:(offset+batch_size), :]
		batch_labels = Ytr[offset:(offset+batch_size)]
		Xt = batch_data
		Yt = batch_labels
		train_step.run(feed_dict={images: Xt, labels: Yt, keep_prob: 1, learning_rate : learnRate})

		# Measure accuracy every 10th iteration
		if i % 10 == 0 :
			train_accuracy = accuracy.eval(feed_dict={images: Xt, labels: Yt, keep_prob: 1.0, learning_rate : learnRate})
			val_accuracy = accuracy.eval(feed_dict={images: Xv, labels: Yv, keep_prob: 1.0, learning_rate : learnRate})
			loss = softmaxLoss.eval(feed_dict={images: Xv, labels: Yv, keep_prob: 1.0, learning_rate : learnRate})
			print "Step %d: Training Accuracy: %.2f , Validation Accuracy : %.2f, loss: %.2f" % (i, train_accuracy*100, val_accuracy*100, loss)
			lossHistory.append(loss)
			accHistory.append(val_accuracy)
			sys.stdout.flush()
	

	""" Finally: Run the trained network on test data
		Note: The system freezes with all the test data at once, so try 2000 of them.
	"""
	test_accuracy = accuracy.eval(feed_dict={images: Xte[:2000], labels: Yte[:2000], keep_prob: 1.0})
	print "Test accuracy %.2f" % test_accuracy
	#print lossHistory
	#print accHistory
	sys.stdout.flush()
