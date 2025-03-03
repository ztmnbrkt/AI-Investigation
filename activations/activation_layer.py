
class ActivationLayer():
    """
    Base class for all activation functions/layers, to be implemented by subclasses.
    """
    def forward(self, inputs):
        ...
    def backward(self, learning_rate, output_error):
        ...
