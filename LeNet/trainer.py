import numpy as np
from copy import deepcopy

from metrics import multiclass_accuracy
from layers import softmax, cross_entropy_loss, softmax_with_cross_entropy

import cupy as cp
from tqdm.notebook import tqdm

from time import time
import os

class Dataset:
    ''' 
    Utility class to hold training and validation data
    '''
    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y

        
class Trainer:
    '''
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    '''
    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-3,
                 learning_rate_decay=1.0):
        '''
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        batch_size, int - batch size
        learning_rate, float - initial learning rate
        learning_rate_decal, float - ratio for decaying learning rate
           every epoch
        '''
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay

        self.optimizers = None

    def setup_optimizers(self):
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)

    def compute_accuracy_and_loss(self, X, y):
        '''
        Computes accuracy on provided data using mini-batches
        '''
        indices = np.arange(X.shape[0])

        sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        batches_indices = np.array_split(indices, sections)
        
        pred = np.zeros_like(y)
        outs = np.zeros((y.shape[0], 10))
        
        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            batch_X_gpu = cp.asarray(batch_X)

            out_batch = self.model.forward(batch_X_gpu).get()
            pred_batch = np.argmax(out_batch, axis=1)
            pred[batch_indices] = pred_batch
            outs[batch_indices] = out_batch
            
            

        loss, grad = softmax_with_cross_entropy(cp.asarray(outs), y)
        
            

        return multiclass_accuracy(pred, y) , loss 
        
    def fit(self):
        '''
        Trains a model
        '''
        if self.optimizers is None:
            self.setup_optimizers()
        
        num_train = self.dataset.train_X.shape[0]

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_val_loss = 9999999
        mempool = cp.get_default_memory_pool()
        for epoch in range(self.num_epochs):
            start_time = time()
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_losses = cp.zeros(len(batches_indices))
            correct = 0
            self.model.training_mode[0] = True
            for batches_id, batch_indices in tqdm(enumerate(batches_indices)):
                s=time()

                batch_X = self.dataset.train_X[batch_indices]
                batch_y = self.dataset.train_y[batch_indices]
                batch_y_gpu = cp.asarray(batch_y)
                
                batch_X_gpu = cp.asarray(batch_X)

                for param in self.model.params().values():
                    param.grad.fill(0)

                out = self.model.forward(batch_X_gpu)

                loss, grad = softmax_with_cross_entropy(out, batch_y)

                self.model.backward(grad)


                prediction = cp.argmax(out, axis=1)
                correct += cp.sum(prediction == batch_y_gpu)

                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    optimizer.update(param.value, param.grad, self.learning_rate)

                
                batch_losses[batches_id] = loss
                mempool.free_all_blocks()


            correct = correct.get()

            self.learning_rate *= self.learning_rate_decay
            
            ave_loss = np.mean(batch_losses.get())
            self.model.training_mode[0] = False
            val_accuracy, val_loss = self.compute_accuracy_and_loss(self.dataset.val_X,self.dataset.val_y)
            
            train_accuracy = correct/len(self.dataset.train_X)


            print(f'Epoch {epoch+1}  Train loss: {batch_losses[-1]:.5f}, Train accuracy: {train_accuracy:.5f}, Val loss: {val_loss:.5f}, val accuracy: {val_accuracy:.5f}, time: {(time() - start_time) :.2f}')
            
            # os.mkdir(f'model_{epoch+1}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f'New best val loss: {val_loss}')
                for param_name, param in self.model.params().items():
                    cp.save(f'best_model/{param_name}.npy', param.value)

            loss_history.append(ave_loss)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)

        return loss_history, train_acc_history, val_acc_history

            


        

                
        
