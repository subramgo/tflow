"""
Auto Encoders



input - encoder-hlayer1 - encoder-hlayer2 - decoder-hlayer-1 - decoder-hlayer-2


Gopi Subramanian
13-March-2016
"""

import tensorflow as tf
import numpy as np 
from sklearn.datasets import make_regression
from tqdm import *


batch_size  = 500
no_features = 500
no_batches = 10

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


### Auto encoder parameters
n_hidden_1 = 250
n_hidden_2 = 50
n_input    = no_features

### Input place holder
x = tf.placeholder(tf.float32, shape = [batch_size,no_features])

weights = {
	'encoder_w_hidden_1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_w_hidden_2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_w_hidden_1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_w_hidden_2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))

}


biases = {
	'encoder_b_1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b_2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b_1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b_2': tf.Variable(tf.random_normal([n_input]))

}

def encoder(input):
	layer_1 = tf.nn.sigmoid( tf.add( tf.matmul(input, weights['encoder_w_hidden_1']), biases['encoder_b_1']))
	layer_2 = tf.nn.sigmoid( tf.add( tf.matmul(layer_1, weights['encoder_w_hidden_2']), biases['encoder_b_2']))

	return layer_2

def decoder(input):
	layer_1 = tf.nn.sigmoid( tf.add( tf.matmul(input, weights['decoder_w_hidden_1']), biases['decoder_b_1']))
	layer_2 = tf.nn.sigmoid( tf.add( tf.matmul(layer_1, weights['decoder_w_hidden_2']), biases['decoder_b_2']))

	return layer_2


encoder_node = encoder(x)
decoder_node = decoder(encoder_node)




y_predicted = decoder_node
y_actual = x 

cost = tf.reduce_mean(tf.square(y_predicted - y_actual))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

epochs = 1000
# Train the model
for epoch in tqdm(range(epochs)):
	#if epoch%100 == 0:
	#	print "Number of epochs completed {}".format(epoch)
	start = 0
	end   = batch_size
	for i in range(no_batches):
		batch = batch_generator(start, end)
		start = end 
		end = end + batch_size
		in_data = batch[0]
		np.random.shuffle(in_data) # Shuffle the data
		feed_dict = {x:in_data}
		session.run(optimizer, feed_dict = feed_dict)


# evaluate training accuracy
curr_loss  = session.run([cost], {x:batch_generator(0, batch_size)[0]})
print
print "############################   Goodness of the model ########################################"
print
print "Loss = {}".format(curr_loss)


## Another nosiy dataset
X_noisy, y  = make_regression(n_samples = 500, n_features = 500, n_targets =1 , noise = 0.25)
curr_loss  = session.run([cost], {x:X_noisy})
print "Loss = {}".format(curr_loss)


session.close()



