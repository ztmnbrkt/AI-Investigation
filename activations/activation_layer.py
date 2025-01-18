from layers import layer
from abc import abstractmethod

class ActivationLayer(layer.Layer):
    """
    Base class for all activation functions/layers, to be implemented by subclasses.
    """
    def __init__(self, activation, activation_prime):
        super.__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    @abstractmethod
    def forward():
        ...

    @abstractmethod
    def backward():
        ...
