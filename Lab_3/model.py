import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, DropoutLayer
)


class ConvNet:
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        # AlexNet
        self.layers = []
        self.layers.append(ConvolutionalLayer(1, 64, filter_size=11, stride=4, padding=2))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=3, stride=2))

        self.layers.append(ConvolutionalLayer(64, 192, filter_size=5, padding=2))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=3, stride=2))

        self.layers.append(ConvolutionalLayer(192, 384, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(ConvolutionalLayer(384, 256, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(ConvolutionalLayer(256, 256, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        
        self.layers.append(MaxPoolingLayer(pool_size=3, stride=2))

        self.layers.append(Flattener())
        self.layers.append(FullyConnectedLayer(256*6*6, 4096))
        self.layers.append(ReLULayer())
        # self.layers.append(DropoutLayer(0.5))

        self.layers.append(FullyConnectedLayer(4096, 4096))
        self.layers.append(ReLULayer())
        # self.layers.append(DropoutLayer(0.5))
        self.layers.append(FullyConnectedLayer(4096, n_output_classes))



#         self.layers.append(ConvolutionalLayer(input_shape[2], conv1_channels, filter_size=3, padding=1))
#         self.layers.append(ReLULayer())
#         self.layers.append(MaxPoolingLayer(pool_size=4, stride=4))

#         self.layers.append(ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1))
#         self.layers.append(ReLULayer())
#         self.layers.append(MaxPoolingLayer(pool_size=4, stride=4))

#         self.layers.append(Flattener())
#         self.layers.append(FullyConnectedLayer(7840, n_output_classes))
        


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
            # print('cur', X.shape)
            X = layer.forward(X)
            if np.isnan(X).sum() > 0:
                print(X.shape, 'out nan', layer.name)
            # print('large forward', (np.abs(X) > 10**6).sum()/X.size, layer.name)

        loss, grad = softmax_with_cross_entropy(X, y)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            # print('large backprop', (np.abs(grad) > 10**6).sum()/grad.size, layer.name)


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
