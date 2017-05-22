import tensorflow as tf 
import numpy as np 
from tqdm import *

X =  np.loadtxt('./data/speech-orig.txt', delimiter=',')
y =  np.loadtxt('./data/speech-labels.txt')


X_inline = X[np.where(y == 0.0)[0]]
n,p = X_inline.shape
X_outliers = X[np.where(y == 1.0)[0]]


### Auto encoder parameters
n_hidden_1 = 250
n_hidden_2 = 50
n_input    = X_inline.shape[1]

### Input place holder
x = tf.placeholder(tf.float32, shape = [None,n_input])

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
loss_dist = tf.abs(y_predicted - y_actual)

cost = tf.reduce_mean(tf.square(y_predicted - y_actual))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

epochs = 1000
batch_size = 100
# Train the model
for epoch in tqdm(range(epochs)):
	np.random.shuffle(X_inline) # Shuffle the data
	feed_dict = {x:X_inline}
	session.run(optimizer, feed_dict = feed_dict)


# evaluate training accuracy
curr_loss  = session.run([cost], {x:X_inline})
print
print "############################   Goodness of the model ########################################"
print
print "Loss = {}".format(curr_loss)

model_saver = tf.train.Saver()
model_saver.save(session, "./model/model01.ckpt")

loss_distribution = session.run(loss_dist,{x:X_inline} )

print loss_distribution


## Another nosiy dataset
#X_noisy, y  = make_regression(n_samples = 500, n_features = 500, n_targets =1 , noise = 0.25)
#curr_loss  = session.run([cost], {x:X_noisy})
#print "Loss = {}".format(curr_loss)


session.close()

