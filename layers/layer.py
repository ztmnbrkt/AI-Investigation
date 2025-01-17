from abc import ABC, abstractmethod 

class Layer(ABC):
    """
    Base calss for all network layers.
    """
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        """
        Abstract method performing the forward pass for this layer, to be implemented by subclasses.

        Args:
            input: The input data to the layer

        Returns:
            output: The data transformed by this layer.

        """
        ...

    @abstractmethod
    def backward(self, output):
        """
        Abstract method performing the backward pass for this layer, to be implemented by subclasses.

        Args:
            output: The data to be passed.

        Returns:
            input: The data transformed by this layer, what would be the "input" data.

        """
        ...