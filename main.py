# Artificial Intelligence, an investiation - Zaid barakat.
# v0.1
## testing routines
import numpy as np
from activations import relu, softmax

X = [50, 30, 20]
Z = [28, 0, -28]

a1 = relu.ReLU()
a2 = softmax.SoftMax()

print(a1.forward(X))
print(a2.forward(X))
