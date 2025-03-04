import numpy as np
from activations import relu, softmax
from layers import layer_dense

X = np.array([[50, 30, 20],
     [10, 20, 30],
     [20, 20, 20]])

Z = np.array([[0, 0, -28],
     [0, 0, 0],
     [0, 0, -25]])

Batch = np.array([
    [[50, 30, 20],
     [10, 20, 30],
     [20, 20, 20]], 
     
    [[ 0,  0,-28],
     [ 0, 12,  0],
     [ 0,  0,-25]]])

a1 = relu.relu.ReLU()
a2 = softmax.softmax.Softmax()

print(a1.forward(Batch))
print(a2.forward(Batch))
