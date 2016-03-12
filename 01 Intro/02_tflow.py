"""
02_tflow.py

1. Variables
2. Create and run a session 
3. Helper function for Variable creation.
4. Saving and Restoring variables

Demonstrate a simple operation using tensor variable  
and matrix and addition operation.

Gopi Subramanian
12-March-2016
"""

import tensorflow as tf 
import numpy as np 

const_matrix = tf.constant(np.asarray([[12,13],[20,30]],dtype=np.float32))
# Variable should be initialzed as const_matrix
a_matrix     = tf.Variable( const_matrix , name = "a_matrix" )
# A matrix of ones
one_matrix   = tf.Variable(tf.ones([2,2], dtype=tf.float32), name = "one_matrix")

add_op_1     = a_matrix + one_matrix
add_op_2     = add_op_1 + one_matrix


# Initialize variables
init = tf.initialize_all_variables()
# Session
session = tf.Session()
session.run(init)
result = session.run(add_op_2)


session.close()

print result




