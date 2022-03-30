import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, DropoutLayer
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        # AlexNet
        self.layers = []
        self.layers.append(ConvolutionalLayer(input_shape[2], conv1_channels, filter_size=11, stride=4))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=3, stride=2))

        self.layers.append(ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=5, padding=2))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=3, stride=2))

        self.layers.append(ConvolutionalLayer(conv2_channels, conv2_channels, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(ConvolutionalLayer(conv2_channels, conv2_channels, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(ConvolutionalLayer(conv2_channels, conv2_channels, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        
        self.layers.append(MaxPoolingLayer(pool_size=3, stride=2))

        self.layers.append(Flattener())
        self.layers.append(FullyConnectedLayer(25000, 1000))
        self.layers.append(ReLULayer())
        self.layers.append(DropoutLayer(0.5))

        self.layers.append(FullyConnectedLayer(1000, 1000))
        self.layers.append(ReLULayer())
        self.layers.append(DropoutLayer(0.5))
        self.layers.append(FullyConnectedLayer(1000, n_output_classes))
        


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for param in self.params().values():
            param.grad = 0

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        for layer in self.layers:
            # print(X.shape)
            X = layer.forward(X)
        loss, grad = softmax_with_cross_entropy(X, y)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        for layer in self.layers:
            X = layer.forward(X)
        pred = np.argmax(X, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for layer_num, layer in enumerate(self.layers):
            for param_name, param in layer.params().items():
                result[param_name + '_' + str(layer_num)] = param

        return result
