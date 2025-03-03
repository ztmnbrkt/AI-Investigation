import numpy as np
from activations import activation_layer

class Sigmoid(activation_layer.ActivationLayer):
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
    
    def backward(self, output_error, learning_rate):
        sigmoid_input = self.forward(self.inputs)
        input_error = output_error * sigmoid_input * (1-sigmoid_input)
        return input_error
    