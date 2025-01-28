from activation_layer import ActivationLayer
import numpy as np

class SoftMax(ActivationLayer):
    """
    Softmax activation: turns outputs of hidden layers into probabillities for each classifier 
    (output or estimation).
    """
    def __init__(self, NumberOfClasses):
        super.__init__()
        self.NumberOfClasses = NumberOfClasses
    
    def forward(self, inputs):