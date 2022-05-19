import numpy as np
import cupy as cp
from time import time


from layers import (
    FullyConnectedLayer, ReLULayer, 
    ConvolutionalLayer, MaxPoolingLayer, Flattener, Reshaper, TransposedConvolutionalLayer,
    softmax_with_cross_entropy, l2_regularization, SigmoidLayer, LeakyReLULayer, DropoutLayer, TanhLayer
)

from metrics import multiclass_accuracy
from tqdm import tqdm


class Discriminator:
    def __init__(self):
        self.layers = []
        self.training_mode = [True]
        
        self.layers.append(Flattener())
        # self.layers.append(DropoutLayer(0.2, self.training_mode))
        self.layers.append(FullyConnectedLayer(28*28*1, 1024))
        self.layers.append(LeakyReLULayer(0.2))
        # self.layers.append(DropoutLayer(0.2, self.training_mode))
        self.layers.append(FullyConnectedLayer(1024, 512))
        self.layers.append(LeakyReLULayer(0.2))
        # self.layers.append(DropoutLayer(0.2, self.training_mode))
        self.layers.append(FullyConnectedLayer(512, 512))
        self.layers.append(LeakyReLULayer(0.2))
        self.layers.append(FullyConnectedLayer(512, 1))
        self.layers.append(LeakyReLULayer(0.2))
        

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
            # print(X.shape, layer.name)

        return X
    
    def backward(self, grad):
        # print("BACKWARD")
        for layer in reversed(self.layers):
            # print(layer.name, grad.sum())
            grad = layer.backward(grad)
        return grad

        
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

            
            

class GeneratorModel:
    def __init__(self):
        self.layers = []
        self.training_mode = [True]
        
        # VER 3
        self.layers.append(FullyConnectedLayer(100, 256))
        self.layers.append(LeakyReLULayer(0.2))
        self.layers.append(FullyConnectedLayer(256, 512))
        self.layers.append(LeakyReLULayer(0.2))
        self.layers.append(FullyConnectedLayer(512, 1024))
        self.layers.append(LeakyReLULayer(0.2))
        self.layers.append(FullyConnectedLayer(1024, 28*28*1))
        self.layers.append(Reshaper((28, 28, 1)))
        self.layers.append(SigmoidLayer())
                       
                       
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
            # print(X.shape)
            if cp.any(cp.isnan(X)):
                print(layer.name)
                return None

        return X
    
    def backward(self, grad):
        # print("BACKWARD")
        for layer in reversed(self.layers):
            # print(layer.name, grad.mean())
            grad = layer.backward(grad)
            
        return 0
            
        # self.reg = 0.0001
        # l2 = 0
        # l2_loss = 0
        # for layer in reversed(self.layers):
        #     grad = layer.backward(grad)
        #     grad_l2 = 0
        #     for params in layer.params():
        #         param = layer.params()[params]
        #         loss_d, grad_d = l2_regularization(param.value, self.reg)
        #         param.grad += grad_d
        #         l2 += loss_d
        #     grad += grad_l2
        # l2_loss +=l2
        # return l2_loss

        
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

            
            
