from activation_layer import ActivationLayer
import numpy as np

class SoftMax(ActivationLayer):
    """
    Softmax activation: turns outputs of hidden layers into probabillities for each classifier, 
    simulatneously retaining the meaning of all possible input values.  
    """
    def __init__(self, NumberOfClasses):
        super.__init__()
        self.NumberOfClasses = NumberOfClasses
    
    def forward(self, inputs):
        exponential_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probalbillities = exponential_values / np.inputs()

        
        ...
    
    def backward(self, outputs):
        ...
    