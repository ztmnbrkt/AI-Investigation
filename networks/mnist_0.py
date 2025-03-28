# Artificial Intelligence, an investiation - Zaid barakat.
# v0.1
## testing routines

from keras import datasets
import numpy as np
from layers import layer_dense
from activations import relu
from activations import softmax

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
# Batch the datasets
def batch(input, num_batches):
    return np.array(np.array_split(input, num_batches))
    
    

network = [layer_dense.LayerDense(28**2, 50), 
           relu.ReLU(),
           layer_dense.LayerDense(50, 10),
           softmax.SoftMax()]

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

X = np.array(batch(x_train, 100))
batches = X[0]
image = x_train[np.random.randint(60000), :, :]
predict(network, image)

print("completed")
