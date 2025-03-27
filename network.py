from keras import datasets
import numpy as np
from layers import layer_dense, convolutional
from activations import relu, softmax, error_functions, sigmoid

class Network():
    def __init__(self, network=[], activaiton=error_functions.BinaryCrossEntropy()):
        self.network = network
        self.activation = activaiton

    def add_layer(self, layer):
        self.network.append(layer)
    
    def predict_single(self, input):
        output = input # Avoids writing another loop for the inputs
        for layer in self.network:
            output = layer.forward(output)
        #print(f"Prediction: {np.argmax(output)}")
        return output

    def train(self, x_train, y_train, epochs=100, batches=0, learning_rate=0.1):
        """
        Initiates network training loop.
        
        Args:
            x_train (list or numpy.ndarray): The input training data.

            y_train (list or numpy.ndarray): The target training data.

            epochs (int, optional): The number of training iterations. Defaults to 100.

            batches (int, optional): The number of batches to divide the data into. 

            If 0, the entire dataset is used for each epoch. Defaults to 0.

            learning_rate (float, optional): The step size for updating weights during 
            backpropagation. Defaults to 0.01.
        Returns:
            None
        """
        if batches !=0:
            pass
        else:
            for e in range(epochs):
                error = 0
                for x, y in zip(x_train, y_train):
                    ## Forward pass
                    output = self.predict_single(x)

                    ## Error calculation
                    error += self.activation.forward(y, output)
                    error /= len(x_train) # averaging out error

                    ## Backward pass
                    error_rate = self.activation.backward(y, output)
                    for layer in reversed(self.network):
                        error_rate = layer.backward(error_rate, learning_rate)
            print(f"Epoch: {e}, Error: {error}")

    def pre_process(self, x, y, classes=10, limit=100):
        
        i, h, w = np.array(x).shape
        all_indices = []
        
        ## Find all label indices upto a limit
        for label in range(classes):
            class_indices = np.where(y == label)[0][:limit]
            all_indices.append(class_indices)
        
        # Collapse to single array
        all_indices= np.hstack(all_indices)

        # Shuffle
        all_indices = np.random.permutation(all_indices)

        # Select data
        x, y = x[all_indices], y[all_indices]

        ## Reshape input to [number_samples, x_shape], normalising to avoid overflow
        x = x.reshape(len(x), 1, h, w)
        #x = x.astype(float) /255

        ## OneHot encode labels, reshape to [number_samples, classes, label]
        y = np.eye(classes)[y] 
        y = y.reshape(len(y), classes, 1)
        
        return x, y
