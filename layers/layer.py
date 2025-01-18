from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Base class for all network layers.
    """
    def __init__(self):
        self.inputs = None
        self.outputs = None

    @abstractmethod
    def forward(self, inputs):
        """
        Abstract method performing the forward propagation for this layer given an input, to be implemented by subclasses.
        """
        ...

    @abstractmethod
    def backward(self, output_error, learning_rate):
        """
        Abstract method performing the backward propagation for this layer, to be implemented by subclasses.
        
        Args:
            output_error: how "wrong" the network is, used in gradient calcualtions.
            learning_rate: a scaler for gradient calculations effect.
        """
        ...