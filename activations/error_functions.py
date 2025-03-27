import numpy as np

class BinaryCrossEntropy():
    def forward(self, y_true, y_predicted):
        ## Clip y_predicted to avoid log(0) or log(1)
        epsilon = 1e-8  # Small value to prevent numerical instability
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        
        return np.mean(-y_true * np.log(y_predicted) - (1 - y_true) * np.log(1 - y_predicted))
    
    def backward(self, y_true, y_predicted):
        ## Clip y_predicted to avoid division by zero
        epsilon = 1e-8  # Small value to prevent numerical instability
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
    
        return (1 - y_true) / (1 - y_predicted) - (y_true / y_predicted) / np.size(y_true)
                
class CategoricalCrossEntropy:
    def forward(self, y_true, y_predicted):
        # Clip predictions to avoid log(0)
        epsilon = 1e-10
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        # Compute categorical cross-entropy
        return -np.sum(y_true * np.log(y_predicted)) / y_true.shape[0]

    def backward(self, y_true, y_predicted):
        # Clip predictions to avoid division by zero
        epsilon = 1e-10
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        # Compute gradient
        return -y_true / y_predicted