import numpy as np
import cupy as cp
from time import time


from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
)

from metrics import multiclass_accuracy
from tqdm import tqdm


class ConvNet:
    def __init__(self):
        self.layers = []
        self.training_mode = [True]

        
        # LeNet-5 
        self.layers.append(ConvolutionalLayer(1, 6, filter_size=5, padding=2))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=2, stride=2))

        self.layers.append(ConvolutionalLayer(6, 16, filter_size=5))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=2, stride=2))

        self.layers.append(Flattener())
        self.layers.append(FullyConnectedLayer(400, 120))
        self.layers.append(ReLULayer())

        self.layers.append(FullyConnectedLayer(120, 84))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(84, 10))

    


    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        
    def predict_accuracy(self, X, y, batch_size):
        self.training_mode[0] = False
        indices = np.arange(X.shape[0])

        sections = np.arange(batch_size, X.shape[0], batch_size)
        batches_indices = np.array_split(indices, sections)
        
        pred = np.zeros_like(y)
        probs = np.zeros((y.shape[0], 10))
        
        for batch_indices in tqdm(batches_indices):
            batch_X = X[batch_indices]
            batch_X_gpu = cp.asarray(batch_X)

            out_batch = self.forward(batch_X_gpu).get()
            pred_batch = np.argmax(out_batch, axis=1)
            pred[batch_indices] = pred_batch

            

        return multiclass_accuracy(pred, y)

    def params(self):
        result = {}
        for layer_num, layer in enumerate(self.layers):
            for param_name, param in layer.params().items():
                result[f'{param_name} {layer.name}_{layer_num}'] = param

        return result
    
    def load_params(self, folder):
        for param_name, param in self.params().items():
            param.value = cp.load(f'{folder}/{param_name}.npy')
