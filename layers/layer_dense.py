from layer import Layer
import numpy as np

class LayerDense(Layer):
    """
    The fully connected dense layer, accepts inputs X and returns outputs Y
    """
    def __init__(self, input_size, output_size):
        """
        Initialises the layer, generating random weight and bias matrices W and B based on input and output sizes.
        Args:
            input_size: Number of input neurones
            output_size: Number of output neurones
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5
        

    def forward(self, inputs):
        """
        Calculates the dot product of X, W, then adding B.
        
        Args:
            inputs: Data to process
        
        Returns:
            self.outputs: Processed data
        """
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.outputs) + self.biases
        return self.outputs
    
    def backward(self, output_error, learning_rate):
        """
        Calculates input error dE/dX, weights error dE/dW and bias error dE/dB for this layers output error dE/dY; 
        Adjusts W and B accrodingly.
        
        Args:
            output_error: dE/dY, the error of this output
            learning_rate: A scalar for the effect of this propogation, a
        Returns:
            input_error: The calculated layer input error, dE/dX
        """
        input_error = np.dot(output_error, self.weights.T) #T means the matrix's transposition, see docs for more.
        weights_error = np.dot(self.inputs.T, output_error)
        # dE/dB = output_error, redundant.

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error