import numpy as np
from layers import layer

class Concvolutional(layer.Layer):
    """
    The Convolutional layer, handles batches and supports tensors only.
    """
    def __init__(self, input_shape, padding = 0, stride = 1, kernel_size = 3, num_kernels = 1): 
        self.input_shape = input_shape 
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        self.kernels = np.random.randn(num_kernels, input_shape[0], kernel_size, kernel_size) # Initialising kernels
        self.biases = np.zeros(self.kernels.shape[0], 1) # Generating biases, initially 0, one per kernel
        ...

    def forward(self, inputs):
        """
        Computes the convolution of inputs X and 
        """
        self.inputs = np.pad(inputs, ((0, 0),(0, 0),
                             (self.padding, self.padding), (self.padding, self.padding)),
                             mode='constant')
        
        batch_size, channel, height, width = self.inputs.shape
        output_height = (height - self.kernel_size) + 1
        output_width = (width - self.kernel_size) + 1
        
        # Initialising output variable
        self.output = np.zeros((batch_size, self.num_kernels, output_height, output_width))
        
        # Looping through input batch to complete convolution:
        for i in range(batch_size):
            for k in range(self.num_kernels):
                for h in range(output_height):
                    for w in range(output_width):
                        # Marking the region to convolute:
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w + self.kernel_size
                        region = self.inputs[i, :, h_start:h_end, w_start:w_end]
                        # Note: the colon is used to combine the convolution for all 
                        # input channels, keeping the output as a single tensor able
                        # to be used to extrac richer features later in the network.
                        # See "Spatial Locations" in docs.
                        self.output[i, k, h, w] = np.sum(region*self.kernels[k]) + self.biases[k]    
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Note that, whilst writing this section of the program, I found it easier to follow
        the "semi-mathematical" notation I have used here, exhanging "output_error" for dY.
        """
        dX = np.zeros_like(self.kernels) # Input error
        dK = np.zeros_like(self.kernels) # Kernel error
        dB = np.sum(output_error, axis=(0, 2, 3)).reshape(self.biases.shape) # Bias error

        # Parsing output error:
        batch_size, channel, output_height, output_width = output_error.shape
        
        # Looping through in input to complete backprop.:
        for i in range(batch_size):
            for k in range(self.num_kernels):
                for h in range(output_height):
                    for w in range (output_width):
                        # Marking region:
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        region = self.inputs[i, :, h_start:h_end, w_start:w_end]

                        # Calculating gradients per spatial location
                        dK[k] += output_error[i, k, h, w] * region
                        dX[i, :, h_start:h_end, w_start:w_end] += output_error[i, k, h, w] * self.kernels[k]
        
        # Again, lack of use of dimention 1 is for Spatial Locations. See docs.
        # The reshape function avoids shape-related issues.
        
        # Adjusting learnable params.:
        self.kernels -= learning_rate * dK
        self.biases -= learning_rate * dB

        # Returns input error, removing padding.
        return dX[:, :, self.padding:-self.padding, self.padding:-self.padding]