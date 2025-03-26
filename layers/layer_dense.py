import numpy as np
from layers import layer

class LayerDense(layer.Layer):
    """
    The fully connected dense layer. 
    Supports and adapts to matrices and tensors.
    """
    def __init__(self, input_size, output_size):
        """
        Initialises the layer, generating random weight and bias matrices W and B.
        Args:
            input_size: Number of input neurones
            output_size: Number of output neurones
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5
        

    def forward(self, input):
        """
        Calculates the dot product of inputs and weights, adding biases.
        Auto-flattens 4D input.
        
        Args:
            inputs: Data to process, shape 
            (batch_size, input_size) OR (batch_size, channel, height, width)
        
        Returns:
            Processed data, shape (batch_size, output_size)
        """
        input = np.array(input)
        self.original_shape = np.array(input.shape) # Retaining original shape
        
        ## Autoflatten image
        if input.ndim == 2:
            input = input.reshape(1, -1)

        ## Autoflatten 4D
        if input.ndim == 4:
            batch_size, channel = input.shape[0], input.shape[1]
            input = input.reshape((batch_size, channel, -1))
            ...

        if input.ndim == 3:
            channel = input.shape[0]
            input = input.reshape((channel, -1))
        self.input = input

        # Forward pass
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    
    def backward(self, output_error, learning_rate):
        """
        backward pass with gradeint reshape to match input shape if it was 4D.

        Args:
            output_error: Gradient of the error with respect to the output
            learning_rate: A scalar for the effect of this propogation
        Returns:
            gradent of the error with respect to the input
        """
        ## Calculating gradients
        dW = np.dot(self.input.T, output_error)
        dB = np.sum(output_error, axis=0, keepdims=True) # Working in terms of batches.
        dX = np.dot(output_error, self.weights.T) # T means transposition, see docs for more.

        ## Auto-reshape
        dX = dX.reshape(self.original_shape)

        ## Adjusting learnable params:
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB
        return dX