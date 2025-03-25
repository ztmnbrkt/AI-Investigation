from keras import datasets
import numpy as np
from layers import layer_dense, convolutional
from activations import relu, softmax, error_functions

class Network():
    def __init__(self, network=[], activaiton=error_functions.BinaryCrossEntropy()):
        self.network = network
        self.activation = activaiton

    def add_layer(self, layer):
        self.network.append(layer)
    
    def predict_single(self, input):
        output = input # Avoids writing another loop for the inputs
        for layer in self.network:
            output = layer.forward(output)
        #print(f"Prediction: {np.argmax(output)}")
        return output

    def train(self, x_train, y_train, epochs=100, learning_rate=0.01):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                ## Forward pass
                output = self.predict_single(x)

                ## Error calculation
                error += self.activation.forward(y, output)

                ## Backward pass
                error_rate = self.activation.backward(y, output)
                for layer in reversed(self.network):
                    error_rate = layer.backward(error_rate, learning_rate)
        print(f"Epoch: {e}, Error: {error}")

network = Network()
network.add_layer(convolutional.Convolutional([1, 28, 28]))
network.add_layer(layer_dense.LayerDense(28**2, 128))
network.add_layer(relu.ReLU())
network.add_layer(layer_dense.LayerDense(128, 10))
network.add_layer(softmax.SoftMax())

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
network.train(x_train, y_train)
