# Artificial Intelligence, an investiation - Zaid barakat.
# v0.1
## testing routines

from keras import datasets
import numpy as np
from layers import layer_dense, convolutional
from activations import relu, softmax, error_functions
from network import Network


## The keras.io recomended way to load this dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

## Should combine these into a "pre-process data", streamlines process
# Batch the datasets
def batch(input, num_batches):
    return np.array(np.array_split(input, num_batches))
# Adds a colour channel to maintain compatibillity with layers
def add_colour_channel(input):
    if input.ndim == 2:
        return input[:, :, np.newaxis]
    return input[:, :, np.newaxis, :, :]

network = Network(activaiton=error_functions.CategoricalCrossEntropy())
network.add_layer(convolutional.Convolutional([1, 28, 28]))
network.add_layer(layer_dense.LayerDense(28**2, 128))
network.add_layer(relu.ReLU())
network.add_layer(layer_dense.LayerDense(128, 10))
network.add_layer(softmax.SoftMax())

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, y_train = network.pre_process(x_train, y_train, limit=10)
network.train(x_train, y_train, learning_rate=0.001)

print("completed")
