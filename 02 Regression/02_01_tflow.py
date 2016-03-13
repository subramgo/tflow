"""
Simple Linear Regression using
Tensorflow

1. Place Holder for input to regression 
2. Simple Linear Regression with l1 regularization (LASSO)
3. Visualization of model using tensorboard

Gopi Subramanian
12-March-2016
"""


import tensorflow as tf 
import numpy as np 
from sklearn.datasets import make_regression

# Make a regression dataset with coeffiecients
X, y, coeff  = make_regression(n_samples = 5000, n_features = 100, n_targets =1 , noise = 0.05, coef = True)
y = np.array([y]).T

# No of instances n
# No of features  p
n, p = X.shape

# Add bias term
X = np.column_stack([np.ones(n), X])
p = p + 1

# Print coeffiecient generated
#for i, c in enumerate(coeff): print i+1,c


# Linear regression model
# y = W * x + b
with tf.name_scope('model') as scope:
    theta  = tf.Variable(tf.zeros([p,1]), name='coeffiecients')
    alpha  = tf.constant(0.001)

# Summary for tensorboard
theta_hist = tf.histogram_summary("theta", theta)


# Place holders for training data
x  = tf.placeholder(tf.float32, shape = [n,p])
y_ = tf.placeholder(tf.float32, shape = [n,1])

# Linear regression with regularization
# y = theta*x
y_pred = tf.matmul(x, theta)

# Check the shape of predicted y
print y_pred.get_shape()

# Loss function with regularization
# cost = 1/2n (sum ( square (theta*x - y )) ) + alpha * abs(theta) )
with tf.name_scope('model') as scope:
    lasso_term = alpha * (tf.reduce_sum(tf.abs(theta)))
    cost = tf.div( tf.reduce_sum( tf.square( tf.sub ( y_pred, y_ ) ) ) + lasso_term , 2*n ,name = "cost")

# For tensorboard visualizing
cost_summary = tf.scalar_summary("cost", cost)

# Gradient descent learning
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# intialize variables and begin session
init = tf.initialize_all_variables()
session = tf.Session()

# Merge summaries
merged_summary = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('./logs/02Regression', session.graph_def)

session.run(init)


old_training_cost = 0
epochs = 5000
# Start training
for i in range(epochs):
	# Shuffle
    in_data = np.column_stack([X, y])
    np.random.shuffle(in_data)
    y = in_data[:,-1]
    y = np.array([y]).T

    X = in_data[:,0:in_data.shape[1]-1]
    feed_dict = {x:X, y_:y}
    session.run(optimizer, feed_dict = feed_dict)

    result = session.run([merged_summary ,cost], feed_dict = feed_dict)
    summary_str = result[0]
    writer.add_summary(summary_str, i)
    training_cost = result[1]
    if np.abs(training_cost - old_training_cost) < 0.000001:
    	break
    
    if i%500 == 0:
		print "Iteration %d Training cost %0.3f"%(i, training_cost)

    old_training_cost = training_cost

print "Iteration %d Training cost %0.3f"%(i, old_training_cost)

writer.flush()
writer.close()
# close session
session.close()



