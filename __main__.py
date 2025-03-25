# Artificial Intelligence, an investiation - Zaid barakat.
# v0.1
## testing routines

from keras import datasets
import numpy as np
from layers import layer_dense
from activations import relu, softmax, error_functions


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

def predict(network, input):
    output = input # Avoids writing another loop for the inputs
    if output.ndim == 5:
        batches: int = output.shape[0]
        for b in range(batches):
            for layer in network:
                processed_output = layer.forward(output[b])
                if processed_output.shape != output[b].shape:
                    processed_output = processed_output.reshape(output[b].shape)
                output[b] = processed_output
        print(f"Prediction: {output}")
    return output

def train(network, activation, x_train, y_train, batches=128, epochs=100, learning_rate=0.01):
    # Split the training data into batches
    x_batches = batch(x_train, batches)
    y_batches = batch(y_train, batches)

    for e in range(epochs):
        total_error = 0

        for b in range(len(x_batches)):
            x_batch = x_batches[b]
            y_batch = y_batches[b]

            # Forward pass
            outputs = []
            for x in x_batch:
                output = x
                for layer in network:
                    output = layer.forward(output)
                outputs.append(output)

            # Convert outputs to numpy array for batch processing
            outputs = np.array(outputs)

            # Error calculation
            batch_error = 0
            for y, output in zip(y_batch, outputs):
                batch_error += activation.forward(y, output)
            total_error += batch_error

            # Backward pass
            for x, y, output in zip(x_batch, y_batch, outputs):
                error_rate = activation.backward(y, output)
                for layer in reversed(network):
                    error_rate = layer.backward(error_rate, learning_rate)

            print(f"Epoch: {e}, Batch: {b}, Batch Error: {batch_error}")

        print(f"Epoch: {e}, Total Error: {total_error}")

"""
def train(network, activation, x_train, y_train, batches=128, epochs=100, learning_rate=0.01):
    for e in range(epochs):
        error = 0
        output = x_train
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
"""

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
#predict(network, X)

print("completed")
