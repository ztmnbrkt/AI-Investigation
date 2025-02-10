import numpy as np
from activations import relu, softmax

X = [[50, 30, 20],
     [10, 20, 30],
     [20, 20, 20]]
Z = [[28, 0, -28],
     [0, 0, 0],
     [0, 0, -25]]

a1 = relu.ReLU()
a2 = softmax.SoftMax()

print(a1.forward(Z))
print(a2.forward(Z))