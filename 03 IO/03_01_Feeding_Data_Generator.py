"""
Training in batch 
Reading data from a Python function 


1. Feed data from a Python generator
2. Batch gradient descent for ridge regression 
3. Reusing theta calculated in the previous batch 

Gopi Subramanian
13-March-2016
"""

import tensorflow as tf 
import numpy as np 
from sklearn.datasets import make_regression

batch_size  = 500
no_features = 50
no_batches = 10
epochs     = 300


"""A Convienient function to return
training data in batch 
"""
def train_data():
    # Make a regression dataset with coeffiecients
    sample_size = batch_size * no_batches
    X, y  = make_regression(n_samples = sample_size, n_features = no_features, n_targets =1 , noise = 0.05)
    y = np.array([y]).T

    # Add bias term
    X = np.column_stack([np.ones(sample_size), X])

    return (X, y)

input_data = train_data()
X = input_data[0]
y = input_data[1]

def batch_generator(start, end):
	x_batch = X[start:end,:]
	y_batch = y[start:end,:]
	return (x_batch, y_batch)


# Build the graph
# Input placeholders
x  = tf.placeholder(tf.float32, shape = [batch_size, no_features + 1], name = "x")
y_ = tf.placeholder(tf.float32, shape = [batch_size, 1], name = "x")
# Coeffiecients
theta  = tf.Variable(tf.zeros([no_features + 1, 1]), name = "theta" )
alpha  = tf.constant(0.001)
# Regression
# y = theta*x
y_pred = tf.matmul(x, theta)
ridge_term = alpha * (tf.reduce_sum(tf.square(theta)))
cost = tf.div( tf.reduce_sum( tf.square( tf.sub ( y_pred, y_ ) ) ) + ridge_term , 2*batch_size ,name = "cost")
rmse = tf.reduce_mean( tf.square ( tf.sub( y_pred, y_)  ) )

# Gradient descent learning
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# intialize variables and begin session
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# Train the model
for i in range(no_batches):
	print 'Training batch %d'%(i+1)
	start = 0
	end   = batch_size
	batch = batch_generator(start, end)
	start = end 
	end = end + batch_size
	feed_dict = {x:batch[0],y_:batch[1]}
	old_training_cost = 0
	for j in range(epochs):
	    session.run(optimizer, feed_dict = feed_dict)
	    training_cost = session.run(cost, feed_dict = feed_dict)
	    if np.abs(training_cost - old_training_cost) < 0.00001:
	    	print '\tTraining cost at iteration %d is %0.3f'%(j+1, training_cost)
	    	break
	    old_training_cost = training_cost
	print 'Evaluation at batch %d is %0.3f'%(i+1, session.run(rmse, feed_dict = feed_dict ))


# close session
session.close()





