from activation_layer import ActivationLayer
import numpy as np
class ReLU(ActivationLayer):
    """
    Rectified Linear Unit activation - removes negative values from the input data.
    """
    def __init__(self):
        super().__init__()
    
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
        self.outputs = np.max(0, self.inputs)
        return self.outputs
    
    def backward(self, learning_rate, error_rate):
        """
        ReLU's packpropagation is the same as it's forward: return x for all values x greater than 0.
        As this has already calculeted in self.outputs, return self.outputs.
        """
        return self.outputs
        ...
