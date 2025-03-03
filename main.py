# Artificial Intelligence, an investiation - Zaid barakat.
# v0.1
## testing routines

from keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
print("finally omd")