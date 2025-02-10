import numpy as np
from layers import layer

class LayerDense(layer.Layer):
    """
    The fully connected dense layer. Supports and adapts to both 2D matrices and 4D 
    tensors.
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
        

    def forward(self, inputs):
        """
        Calculates the dot product of inputs and weights, adding biases.
        Auto-flattens 4D input.
        
        Args:
            inputs: Data to process, shape 
            (batch_size, input_size) OR (batch_size, channel, height, width)
        
        Returns:
            Processed data, shape (batch_size, output_size)
        """

        # Autoflatten
        self.original_shape = inputs.shape # Retaining original shape
        if inputs.ndim > 2:
            batch_size = inputs.shape[0]
            inputs = inputs.reshape(batch_size, -1) # Flattening

        self.inputs = inputs

        # Forward pass
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    
    def backward(self, output_error, learning_rate):
        """
        backward pass with gradeint reshape to match input shape if it was 4D.

        Args:
            output_error: Gradient of the error with respect to the output
            learning_rate: A scalar for the effect of this propogation
        Returns:
            gradent of the error with respect to the input
        """
        # Calculating gradients
        input_error = np.dot(output_error, self.weights.T) # T means transposition, see docs for more.
        weights_error = np.dot(self.inputs.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True) # Working in terms of batches.

        # Adjusting learnable params.:
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * bias_error
        return input_error