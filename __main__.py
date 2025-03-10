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
assert y_train.shape == (60000, )
assert y_test.shape == (10000, )


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
    if input.ndim == 2:
        return input[:, :, np.newaxis]
    return input[:, :, np.newaxis, :, :]


def predict(network, input):
    output = input # Avoids writing another loop for the inputs
    if output.ndim == 5:
        batches: int = output.shape[0]
        for b in range(batches):
            for layer in network:
                output[b] = layer.forward(output[b])
    print(f"Prediction: {output}")
    return output

## not properly assigning input and output variables
def train(network, activation, x_train, y_train, batches=128, epochs=100, learning_rate=0.01):
    for e in range(epochs):
        error = 0
        output = input
        for b in range(batches):
            for x, y in zip(x_train, y_train):
                # Forward
                for layer in network:
                    output[b] = layer.forward(output[b])
                output = predict(network, x)

                # Error calculation
                error += activation.forward(y, output)
                
                # Backward
                error_rate = activation.backward(y, output)
                for layer in reversed(network):
                    error_rate = layer.backward(error_rate, learning_rate)

            print(f"Prediction: {output}")
            print(f"Epoch: {e}, Batch: {b} Error: {error}")

    ...


'''
## Predict single image
X = np.array([1, 1, 1, 1, 1])
network = [layer_dense.LayerDense(5, 5), 
           relu.ReLU(),
           layer_dense.LayerDense(5, 5),
           softmax.SoftMax()]
predict(network, X)
'''

'''
# Predict dataset
X = np.array(batch(x_train, 100))
X = add_colour_channel(X)
network = [layer_dense.LayerDense(28**2, 50), 
           relu.ReLU(),
           layer_dense.LayerDense(50, 10),
           softmax.SoftMax()]
#image = x_train[np.random.randint(60000), :, :]
predict(network, X)
'''

# train dataset
X = np.array(batch(x_train, 100))
X = add_colour_channel(X)
Y = np.array(batch(y_train, 100))
Y = add_colour_channel(Y)
network = [layer_dense.LayerDense(28**2, 50), 
           relu.ReLU(),
           layer_dense.LayerDense(50, 10),
           softmax.SoftMax()]
#image = x_train[np.random.randint(60000), :, :]
train(network, error_functions.BinaryCrossEntropy(), X, Y, batches=100)

print("completed")
