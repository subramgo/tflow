"""
01_tflow.py

1. Constants
2. Create and run a session 
3. Sequences
4. Random Values

Demonstrate a simple operation using tensor constant 
addition operation.

Gopi Subramanian
11-March-2016
"""

import tensorflow as tf 
import numpy as np 


tf.set_random_seed(100)

# Create a tensor constant
a_constant = tf.constant(20, shape = [1], dtype = tf.float32, name = "a_constant")
# Convert a numpy array to a tensor object
conv_array =  tf.convert_to_tensor(np.arange(1,10,0.2), dtype = tf.float32, name = "conv_array")
# Define a simple addition operation
op_add = tf.add(a_constant, conv_array, name = "op_add")

# Create a graph session
session = tf.Session()
output  = session.run(op_add)

# Close the session
session.close()

# 
print output

# Get info about the constant
print conv_array.name
print
print conv_array.get_shape()
print 
print conv_array.op

print a_constant.name
print
print a_constant.get_shape()
print 
print a_constant.op





