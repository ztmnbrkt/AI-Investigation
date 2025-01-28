from layers import layer
from abc import abstractmethod

class ActivationLayer(layer.Layer):
    """
    Base class for all activation functions/layers, to be implemented by subclasses.
    """
    def __init__(self):
        super.__init__()

    @abstractmethod
    def forward(self, inputs):
        ...

    @abstractmethod
    def backward(self, learning_rate, output_error):
        ...
