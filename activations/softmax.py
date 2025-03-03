import numpy as np
from activations import activation_layer

class SoftMax(activation_layer.ActivationLayer):
    """
    Softmax activation: turns outputs of hidden layers into probabillities for each classifier, 
    simulatneously retaining the meaning of all possible input values.  
    """
    def __init__(self):
        ...
    
    def forward(self, inputs):
        """
        Exponentiates and normalises all inputs.
        Supports and adapts to matrices and tensors.
        """
        ## Autoflatten
        self.original_shape = inputs.shape
        if inputs.ndim > 2:
            batch_size = inputs.shape[0]
            inputs = inputs.reshape(batch_size, -1)

        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # np.max() clips the values so values dont explode.

        probabillities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True) 

        self.output = probabillities
        return self.output
    
    def backward(self, output_error, learning_rate):
        """
        Returns the error of the next function, SoftMax's derrivative is 0.
        """
        return output_error
