"""
04_tflow.py

Segmentation

TensorFlow provides several operations that you can use to perform common math computations on tensor segments. 
Here a segmentation is a partitioning of a tensor along the first dimension, i.e. it defines a mapping from the 
first dimension onto segment_ids. 
The segment_ids tensor should be the size of the first dimension, d0, 
with consecutive IDs in the range 0 to k, where k<d0. 
In particular, a segmentation of a matrix tensor is a mapping of rows to segments.

Gopi Subramanian
12-March-2016
"""

import tensorflow as tf 
import numpy as np 