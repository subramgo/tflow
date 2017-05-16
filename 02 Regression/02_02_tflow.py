"""
Linear Regression using tf.contrib.learn


tf.contrib.learn is a high level TensorFlow Library for Machine Learning.
A point to note is this module is still under development.

Gopi Subramanian
16-May-2017

"""


import tensorflow as tf 
import numpy as np 
from sklearn.datasets import make_regression


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


features = [tf.contrib.layers.real_valued_column("x", dimension = p)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns = features )
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":X},y, batch_size=4, num_epochs=1000)


estimator.fit(input_fn = input_fn, steps =1000)
print(estimator.evaluate(input_fn=input_fn))
