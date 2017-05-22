"""
Simple Linear Regression with l1 regularization (LASSO)

Gopi Subramanian
16-May-2017

"""


import tensorflow as tf 
import numpy as np 
from sklearn.datasets import make_regression
import time

np.random.seed(100)
### Make a regression dataset with coeffiecients
X, y, coeff  = make_regression(n_samples = 1000, n_features = 10, n_targets =1 , noise = 0.05, coef = True, bias = 0.03)
y = np.array([y]).T
n, p = X.shape

### Add bias term
X = np.column_stack([np.ones(n), X])
p = p + 1
print "Created Regression input, X = ({},{}), y = ({}) ".format(n,p,n)


tf.set_random_seed(100)

### Model Parameters
theta = tf.Variable(tf.zeros([p,1]), name='theta')
### Place holders for training data
x  = tf.placeholder(tf.float32, shape = [None,p])
y_ = tf.placeholder(tf.float32, shape = [None,1])

### Loss function with regularization
### cost = 1/2n (sum ( square (theta*x - y )) ) + alpha * abs(theta) )
alpha  = tf.constant(0.001)
lasso_term = alpha * (tf.reduce_sum(tf.abs(theta)))
linear_model = tf.matmul(x, theta) 
cost = tf.div(tf.reduce_sum ( tf.square( tf.subtract(linear_model , y_) )) + lasso_term, 2*n, name = 'cost')

### Optimizer
### Gradient descent learning
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


### Training
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

epochs = 2000
startTime = time.time()
# Start training
for i in range(epochs):
    feed_dict = {x:X, y_:y}
    session.run(optimizer, feed_dict = feed_dict)

print("Time taken: %f" % (time.time() - startTime))

# evaluate training accuracy
curr_theta, curr_loss  = session.run([theta,cost], {x:X, y_:y})
print
print "############################   Goodness of the model ########################################"
print
print "Loss = {}".format(curr_loss)
print "Bias, actual = 0.03, Model = {}".format(curr_theta[0])
for i in range(1,p-1):
    print "Coeffiecients, Actual {} Model {} Diff {}".format(coeff[i-1], curr_theta[i], coeff[i-1] - curr_theta[i])




session.close()



