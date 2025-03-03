import numpy as np
from activations import activation_layer
class ReLU(activation_layer.ActivationLayer):
    """
    Rectified Linear Unit activation - removes negative values from the input data.
    """
    def forward(self, inputs):
        """
        Returns the maximum value between 0 and every input element, replacing negative elements with 0,
        This function allows the network to adapt non-linearly, "rectifying" the linear funciton.

        Args:
            Inputs: input data (matrix, vector, number...)

        Returns:
            output: the transformed data
        """
        self.inputs = inputs
        self.outputs = np.maximum(0, self.inputs)
        return self.outputs
    
    def backward(self, learning_rate, output_error):
        """
        ReLU's packpropagation is the same as it's forward: return x for all values x greater than 0.
        As this has already calculeted in self.outputs, return self.outputs.
        """
        return self.outputs
        ...
