"""
03_tflow.py

Different method to initialize variables
Assert if variables are initialized 
Reduce and Accumulate Functionality
    reduce_sum, reduce_prod 
    reduce_min, reduce_max
    reduce_mean, reduce_all
    reduce_any,
    accumulate_n



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

# Sum of all elements of resultant matrix
matrix_sum_1   = tf.reduce_sum( add_op_1, name = "matrix_sum_1")
matrix_sum_2   = tf.reduce_sum( add_op_2, name = "matrix_sum_2")

# Product of all elements
prod_1 = tf.reduce_prod( add_op_1)

# reduce_min, reduce_max, reduce_mean
# reduce_all - item wise AND operator applied
# reduce_any - item wise OR operator applied

# Element wise sum of matrices of same shape
# shape paramter is inferred
element_sum = tf.accumulate_n([add_op_1, add_op_2])


# Initialize variables
init = tf.initialize_variables([one_matrix, a_matrix])

# Session
session = tf.Session()
session.run(init)

try:
   assert_op = tf.assert_variables_initialized([one_matrix, a_matrix])
   result  = session.run([ element_sum,prod_1,matrix_sum_2, matrix_sum_1])
except tf.errors.FailedPreconditionError:
	print 'Intialize variables before using them, exiting session'

session.close()

print result[0]
print result[1] # Output of reduce_product
# Output of reduce_sum
print result[2]
print result[3]


