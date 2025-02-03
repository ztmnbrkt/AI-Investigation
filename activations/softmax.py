from activation_layer import ActivationLayer
import numpy as np

class SoftMax(ActivationLayer):
    """
    Softmax activation: turns outputs of hidden layers into probabillities for each classifier, 
    simulatneously retaining the meaning of all possible input values.  
    """
    def __init__(self):
        super.__init__()
    
    def forward(self, inputs):
        """
        Calculates SoftMax for all inputs, outputting probalbillities for number of inputs.
        """
        exponential_values = np.exp(inputs - np.max(inputs, axis =1, keepdims = True))
        probalbillities = exponential_values / np.sum(exponential_values, axis = 1, deepdims = True)
        self.output = probalbillities
        return self.output
        ...
    
    def backward(self, ):
        """
        SoftMax's derivative is a Jacobian matrix, it doesnt really have a back propogation.



        Returns 
        """
        ...
    