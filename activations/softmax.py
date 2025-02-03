import numpy as np
from activations import activation_layer

class SoftMax(activation_layer.ActivationLayer):
    """
    Softmax activation: turns outputs of hidden layers into probabillities for each classifier, 
    simulatneously retaining the meaning of all possible input values.  
    """
    def __init__(self):
        ...
    
    def forward(self, inputs):
        """
        Calculates SoftMax for all inputs, outputting probalbillities for number of inputs.
        """
        exponential_values = np.exp(inputs - np.max(inputs, keepdims = True))
        probalbillities = exponential_values / np.sum(exponential_values, keepdims = True) #when adding batches, add param: axis = 1
        self.output = probalbillities
        return self.output
        ...
    
    def backward(self):
        """
        SoftMax's derivative is a Jacobian matrix, it doesnt really have a back propogation.



        Returns 
        """
        ...
    