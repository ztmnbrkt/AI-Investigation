from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Base class for all network layers.

    Args:
        input_size: Size of input array
        output_size: Size of output array
    """
    def __init__(self, input_size, output_size):
        ...

    @abstractmethod
    def forward(self, inputs):
        """
        Abstract method performing the forward propagation for this layer given an input, to be implemented by subclasses.

        Args:
            inputs: Data to process
        """
        self.inputs = inputs
        ...

    @abstractmethod
    def backward(self, output_error, learning_rate):
        """
        Calculates input error dE/dX, weights error dE/dW and bias error dE/dB for this layers output error dE/dY; 
        Adjusts W and B accrodingly. See document for more information.
        
        Args:
            output_error: dE/dY, the error of this output
            learning_rate: A scalar for the effect of this back propogation
        """
        self.output_error = output_error
        self.learning_rate = learning_rate
        ...