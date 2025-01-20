from activation_layer import ActivationLayer
import numpy as np
class ActivationReLU(ActivationLayer):
    """
    Rectified Linear Unit activation - removes negative values from the input data.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self):
        """
        Returns the maximum value between 0 and every input element, replacing negative elements with 0,
        This function allows the network to adapt non-linearly, "rectifying" the linear funciton.

        Args:
            Inputs: input data (matrix, vector, number)

        Returns:
            output: the transformed data
        """
        self.outputs = np.max(0, self.inputs)
        return self.outputs
    
    def backward(self):
        raise NotImplementedError
        ...
