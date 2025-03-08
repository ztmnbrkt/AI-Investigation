# Artificial Intelligence, an investiation - Zaid barakat.
# v0.1
## testing routines

from keras import datasets
import numpy as np
from layers import layer_dense
from activations import relu, softmax, error_functions


## The keras.io recomended way to load this dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


'''
def batch(input):
    batches = input.shape[0] / 100
    output = np.array([100])
    for b in range(int(batches)):
         output[b] = input[(b-1)*100:b*100,:,:] # [b, i, w, h]
    return output
'''

## Should combine these into a "pre-process data", streamlines process
# Batch the datasets
def batch(input, num_batches):
    return np.array(np.array_split(input, num_batches))
# Adds a colour channel to maintain compatibillity with layers
def add_colour_channel(input):
    return input[:, :, np.newaxis, :, :]


def predict(network, input):
    output = input # Avoids writing another loop for the inputs
    for layer in network:
        output = layer.forward(output)
    print(f"Prediction: {output}")

def train(network, error, activation, x_train, y_train, batches, epochs = 100, learning_rate = 0.01):
    for e in range(epochs):
        error = 0
        for b in range(batches):
            for x, y in zip(x_train, y_train):
                # Forward
                output = predict(network, x)
                error += activation.forward(y, output)
                
                # Backward
                error_rate = activation.backward(output)
                for layer in reversed(network):
                    error_rate = layer.backward(error_rate, learning_rate)

            print(f"Prediction: {output}")
            print(f"Epoch: {e}, Batch: {b} Error: {error}")

    ...


'''
## Custom Single Image
X = np.array([1, 1, 1, 1, 1])
network = [layer_dense.LayerDense(5, 5), 
           relu.ReLU(),
           layer_dense.LayerDense(5, 5),
           softmax.SoftMax()]
predict(network, X)
'''

## MNIST Single Image
X = np.array(batch(x_train, 100))
X = add_colour_channel(X)
network = [layer_dense.LayerDense(28**2, 50), 
           relu.ReLU(),
           layer_dense.LayerDense(50, 10),
           softmax.SoftMax()]
#image = x_train[np.random.randint(60000), :, :]
predict(network, X)

print("completed")


