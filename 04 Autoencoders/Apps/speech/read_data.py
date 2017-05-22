import numpy as np 

X =  np.loadtxt('./data/speech-orig.txt', delimiter=',')
y =  np.loadtxt('./data/speech-labels.txt')

print X.shape
print y.shape


from collections import Counter
print Counter(y)

X_inline = X[np.where(y == 0.0)[0]]
X_outliers = X[np.where(y == 1.0)[0]]

print X_inline.shape
print X_outliers.shape

