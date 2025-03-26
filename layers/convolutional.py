import numpy as np
from layers import layer

class Convolutional(layer.Layer):
    """
    The Convolutional layer, handles batches and supports tensors only.
    """
    def __init__(self, input_shape, padding = 1, stride = 1, kernel_size = 3, num_kernels = 1): 
        """
        Initialises the convolutional layer, generating random kernels and biases.
        Input shape should be in the form (channel, height, width).
        """
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        ## Batch code:
        #
        self.kernels = np.random.randn(num_kernels, input_shape[0], kernel_size, kernel_size) # Initialising kernels
        self.biases = np.zeros((self.kernels.shape[0], 1)) # Generating biases, initially 0, one per kernel
        ...

    def forward(self, inputs):
        """
        Computes the convolution of inputs X and 
        """
        ## testing something out - expand to add batch/channel dimensions if input was a single image
        reshape = 0

        if inputs.ndim == 3: ## adding batch dimension
            inputs = np.expand_dims(inputs, 0)
            reshape = 1
        elif inputs.ndim == 2: ## adding batch, channel dimensions
            inputs = np.expand_dims(inputs, 0)
            inputs = np.expand_dims(inputs, 0)
            reshape = 2
        ## test end

        if self.padding > 0:
            self.inputs = np.pad(inputs, ((0, 0), (0, 0),
                                 (self.padding, self.padding), (self.padding, self.padding)),
                                 mode='constant')
        else:
            self.inputs = inputs
        
        batch_size, channel, height, width = self.inputs.shape
        output_height = (height - self.kernel_size) + 1
        output_width = (width - self.kernel_size) + 1
        
        # Initialising output variable
        self.output = np.zeros((batch_size, self.num_kernels, output_height, output_width))
        
        # Looping through input batch to complete convolution:
        # Note: using a library like scipy can greatly improve performance, this is an unoptimised
        #       way of doing things written for the sake of my understanding.
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
        
        ## testing something out - return output without batch dimension if input was a single image
        if reshape == 1 and self.output.shape[0] == 1:
            return self.output[0]
        elif reshape == 2 and self.output.shape[1] == 1:
            return self.output[0, 0]
        ## test end

        return self.output

    def forward_single(self, inputs):
            """
            Computes the convolution of inputs X and 
            """
            self.inputs = inputs # temporary while i figure why batches dont work
            
            self.inputs = np.pad(inputs, (
                                (self.padding, self.padding), (self.padding, self.padding)),
                               mode='constant')
            
            channel = 1 # temporary
            height, width = self.inputs.shape
            output_height = (height - self.kernel_size) + 1
            output_width = (width - self.kernel_size) + 1
            
            # Initialising output variable
            self.output = np.zeros((self.num_kernels, output_height, output_width))
            
            # Looping through input batch to complete convolution:
        
            for k in range(self.num_kernels):
                for h in range(output_height):
                    for w in range(output_width):
                        # Marking the region to convolute:
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w + self.kernel_size
                        region = self.inputs[h_start:h_end, w_start:w_end]
                        # Note: the colon is used to combine the convolution for all 
                        # input channels, keeping the output as a single tensor able
                        # to be used to extrac richer features later in the network.
                        # See "Spatial Locations" in docs.
                        self.output[k, h, w] = np.sum(region*self.kernels[k]) + self.biases[k]    
            return self.output

    def backward(self, output_error, learning_rate):
        """
        Note that, whilst writing this section of the program, I found it easier to follow
        the "semi-mathematical" notation I have used here, exhanging "output_error" for dY.
        """

        ## testing something out - expand output without batch/channel dimensions if input was a single image
        if output_error.ndim == 2:
            output_error= np.expand_dims(output_error, 0)
            output_error = np.expand_dims(output_error, 0)
        if output_error.ndim == 3:
            output_error = np.expand_dims(output_error, 0)
        ## test end

        dX = np.zeros_like(self.inputs, dtype=float) # Input error
        dK = np.zeros_like(self.kernels) # Kernel error
        dB = np.sum(output_error, axis=(0, 2, 3)).reshape(self.biases.shape) # Bias error

        # Parsing output error:
        batch_size, channel, output_height, output_width = output_error.shape
        
        # Looping through in input to complete backprop.:
        for i in range(batch_size):
            for k in range(self.num_kernels):
                for h in range(output_height):
                    for w in range (output_width):
                        # Slicing region
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        ## Ensure slicing ranges match kernel size, extract input region
                        #if (h_end - h_start != self.kernel_size) or (w_end - w_start != self.kernel_size):
                        #    continue  # Skip this iteration if the kernel goes out of bounds

                        region = self.inputs[i, :, h_start:h_end, w_start:w_end]
                        
                        # Calculating gradients per spatial location
                        dK[k] += output_error[i, k, h, w] * region
                        dX[i, :, h_start:h_end, w_start:w_end] += output_error[i, k, h, w] * self.kernels[k]
        
        # Again, lack of use of dimention 1 is for Spatial Locations. See docs.
        # The reshape function avoids shape-related issues.
        
        # Adjusting learnable params.:
        self.kernels -= learning_rate * dK
        self.biases -= learning_rate * dB

        ## testing something out - return input error without batch dimension if input was a single image
        if self.inputs.shape[0] == 1:  # Single image in batch
            if self.inputs.shape[1] == 1:  # Single channel
                return dX[0, 0, self.padding:-self.padding, self.padding:-self.padding]  # Remove batch and channel dimensions
            return dX[0, :, self.padding:-self.padding, self.padding:-self.padding]  # Remove batch dimension only

        ## test end

        # Returns input error, removing padding.
        return dX[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def backward_single(self, output_error, learning_rate):
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
        for k in range(self.num_kernels):
            for h in range(output_height):
                for w in range (output_width):
                    # Marking region:
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = w * self.stride
                    w_end = w_start + self.kernel_size
                    region = self.inputs[:, h_start:h_end, w_start:w_end]

                    # Calculating gradients per spatial location
                    dK[k] += output_error[k, h, w] * region
                    dX[:, h_start:h_end, w_start:w_end] += output_error[i, k, h, w] * self.kernels[k]
        
        # Again, lack of use of dimention 1 is for Spatial Locations. See docs.
        # The reshape function avoids shape-related issues.
        
        # Adjusting learnable params.:
        self.kernels -= learning_rate * dK
        self.biases -= learning_rate * dB

        # Returns input error, removing padding.
        return dX[:, self.padding:-self.padding, self.padding:-self.padding]