from keras import datasets

from layers import layer_dense
from activations import relu
from activations import softmax

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)

network = [layer_dense.LayerDense(28**2, 50), 
           relu.ReLU(),
           layer_dense.LayerDense(28**2, 10),
           softmax.SoftMax()]

def predict(network, input):
    output = input # Avoids writing another loop for the inputs
    for layer in network:
        output = layer.forward(output)
    print("Prediction: {output}")

def train(network, error, activation, x_train, y_train, epochs = 100, learning_rate = 0.01):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # Forward
            output = predict(network, x)
            error += activation.forward(y, output)
            
            # Backward
            error_rate = activation.backward(output)
            for layer in reversed(network):
                error_rate = layer.backward(error_rate, learning_rate)

        print("Epoch: {epochs}, Error: {error}")
    ...

predict(network, x_train)
