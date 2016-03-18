"""
Auto Encoders



Gopi Subramanian
13-March-2016
"""

import tensorflow as tf
import numpy as np 
from sklearn.datasets import make_regression


batch_size  = 500
no_features = 50
no_batches = 10
epochs     = 500

tf.set_random_seed(100)


"""A Convienient function to return
training data in batch 
"""
def train_data():
    # Make a regression dataset with coeffiecients
    sample_size = batch_size * no_batches
    X, y  = make_regression(n_samples = sample_size, n_features = no_features, n_targets =1 , noise = 0.05)
    y = np.array([y]).T


    return (X, y)

input_data = train_data()
X = input_data[0]
y = input_data[1]

def batch_generator(start, end):
	x_batch = X[start:end,:]
	y_batch = y[start:end,:]
	return (x_batch, y_batch)


# Number of auto encoders stacked together
NO_LAYERS         = 1
HIDDEN_LAYER_SIZE = 10
input_layer_size = X.shape[1]



# Build graph
W = tf.Variable(tf.random_uniform([input_layer_size, HIDDEN_LAYER_SIZE] )   , name = "Weight")
b = tf.Variable(tf.zeros([1]))
b_dash = tf.Variable(tf.zeros([1]))

x  = tf.placeholder(tf.float32, shape = [batch_size, input_layer_size])

y_b = tf.matmul(x,W)
y = tf.nn.sigmoid(tf.matmul(x, W) + b)


W_dash = tf.transpose(W)
z = tf.nn.sigmoid( tf.matmul(y, W_dash) + b_dash )
cost = tf.sqrt(tf.reduce_mean(tf.square(x - z)))

learning_rate = 0.001
training = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



# intialize variables and begin session
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

#print session.run(W)

# Train the model
for i in range(no_batches):
	print 'Training batch %d'%(i+1)
	start = 0
	end   = batch_size
	batch = batch_generator(start, end)
	start = end 
	end = end + batch_size
	in_data = batch[0]
	old_training_cost = 0
	for j in range(epochs):
	    np.random.shuffle(in_data) # Shuffle the data
	    feed_dict = {x:in_data}
	    session.run(training, feed_dict = feed_dict)
	    training_cost = session.run(cost, feed_dict = feed_dict)
	    if np.abs(training_cost - old_training_cost) < 0.00001:
	    	print '\tTraining cost at iteration %d is %0.3f'%(j+1, training_cost)
	    	break
	    old_training_cost = training_cost
	print 'Evaluation at batch %d is %0.3f'%(i+1, session.run(cost, feed_dict = feed_dict ))






