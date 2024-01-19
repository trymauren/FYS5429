import numpy as np


w = np.zeros((2,2))
w[0,0] = 2
w[0,1] = 1

w[1,0] = 3
w[1,1] = 4

x = np.zeros((0,2))
print(x.shape)
print(w.shape)
# print(x.T.shape)
print(np.dot(w,x.T))
print(x@w)