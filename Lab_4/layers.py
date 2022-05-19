from calendar import c
from turtle import dot
import numpy as np
import cupy as cp
from time import time


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * cp.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''

    # TODO implement softmax
    if len(predictions.shape) == 2:
        # print('asd')
        s = cp.max(predictions, axis=1)
        e_x = cp.exp(predictions - s[:, cp.newaxis])
        div = cp.sum(e_x, axis=1)
        return e_x / div[:, cp.newaxis]
    else:
        exps = cp.exp(predictions - cp.max(predictions))
        return exps / cp.sum(exps)
    
    

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if type(target_index) == int:
        target_index = cp.asarray([target_index])
    if len(target_index.shape) == 1:
        probs = probs[target_index]
    else:
        probs = probs[range(probs.shape[0]), target_index[:, 0]]
        
    probs = cp.clip(probs, 0.001, 1)

    return -cp.mean(cp.log(probs))


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model preds,
    including the gradient

    Arguments:
      preds, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as preds - gradient of preds by loss value
    """
    if type(target_index) == int:
        index = cp.asarray([target_index])
    else:
        index = target_index.copy()

    if index.ndim == 1 and index.size > 1:
        index = index.reshape(-1, 1)

    prob = softmax(preds.copy())
        
    loss = cross_entropy_loss(prob, index)

    y = cp.zeros_like(preds)

    if len(index.shape) == 1:
        y[index] = 1
    else:
        y[range(y.shape[0]), index[:, 0]] = 1

    
    dprediction = prob - 1 * y
        
    if preds.ndim == 2:
        dprediction = dprediction / preds.shape[0]
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = cp.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.name = 'relu'
        self.indexes = None
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        # print(X)
        
        
        if self.indexes is None or self.indexes.shape[0] != X.shape[0]:
            self.indexes = cp.zeros_like(X, dtype=np.bool)
        
        cp.less(X, 0, out=self.indexes)
            
        result = X
        # result[self.indexes] = 0
        cp.multiply(result, self.indexes, out=result)
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        d_result = d_out
        cp.multiply(d_result, self.indexes, out=d_result)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * cp.random.randn(n_input, n_output))
        self.B = Param(0.001 * cp.random.randn(1, n_output))
        self.X = None
        self.name = 'FC'

    def forward(self, X):

        self.X = X
        res = cp.dot(X, self.W.value) + self.B.value
        
        return res

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """


        self.W.grad = cp.dot(self.X.T, d_out)  # dL/dW = dL/dZ * dZ/dW = X.T * dL/dZ
        self.B.grad = cp.array([cp.sum(d_out, axis=0)])  # dL/dB = dL/dZ * dZ/dB = I * dL/dZ
        gradX = cp.dot(d_out, self.W.value.T)  # dL/dX = dL/dZ * dZ/dX = dL/dZ * W.T
        return gradX

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding=0, stride=1):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            cp.random.randn(filter_size, filter_size,
                            in_channels, 
                            out_channels)
        )

        self.B = Param(cp.zeros(out_channels))

        self.padding = padding
        self.stride = stride
        self.name = 'conv'
        
        self.out = None
        self.grad_X = None
        
        self.X = None
        
        self.slice_X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        out_height = (np.floor(height + 2 * self.padding - self.filter_size)/self.stride + 1).astype(np.int32)
        out_width = (np.floor(width + 2 * self.padding - self.filter_size)/self.stride + 1).astype(np.int32)
        

        if self.out is None or self.out.shape[0] != batch_size:
            self.out = cp.zeros([batch_size, out_height, out_width, self.out_channels])
            
        
        if self.X is None:
            self.X = cp.pad(
                array=X,
                pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode='constant'
                )
        else:
            if self.padding == 0:
                self.X = X
            else:
                self.X[:, self.padding:-self.padding, self.padding:-self.padding, :] = X

            
        slice_W = self.W.value.reshape(-1, self.out_channels)

        if self.slice_X is None:
            self.slice_X = cp.zeros_like(self.X[:, 0:self.filter_size, 0:self.filter_size, :].reshape(batch_size, -1))
            
        for y in range(0, out_height):
            for x in range(0, out_width):
                self.slice_X = cp.reshape(self.X[:, y*self.stride:y*self.stride + self.filter_size, \
                                       x*self.stride:x*self.stride + self.filter_size, :], (batch_size, -1))
                

                self.out[:, y, x, :] = self.slice_X.dot(slice_W) + self.B.value
                
                
        return self.out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape


        slice_W = self.W.value.reshape(-1, self.out_channels)
        
        if self.grad_X is None or not np.array_equal(self.grad_X.shape, self.X) :
            grad_X = cp.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):

                slice_X = self.X[:, y*self.stride:y*self.stride + self.filter_size, x*self.stride:x*self.stride + self.filter_size, :] \
                    .reshape(batch_size, -1)

                self.W.grad += cp.dot(slice_X.T, d_out[:, y, x, :]).reshape((self.filter_size, self.filter_size,
                                                                             self.in_channels, out_channels))

                self.B.grad += cp.sum(d_out[:, y, x, :], axis=0)

                grad_X[:, y*self.stride:y*self.stride + self.filter_size, x*self.stride:x*self.stride + self.filter_size, :] += cp.dot(d_out[:, y, x, :], slice_W.T) \
                    .reshape((batch_size, self.filter_size, self.filter_size, self.in_channels))

                pass
        return grad_X[:, self.padding:height - self.padding, self.padding:width - self.padding, :]

    def params(self):
        return {'W': self.W, 'B': self.B}






class MaxPoolingLayer:
    def __init__(self, pool_size, stride=1):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.name = 'maxpooling'
        
        self.out = None
        self.result = None
        self.tmp_result =None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        if self.out is None or not np.array_equal(self.out.shape, [batch_size, out_height, out_width, channels]):
            self.out = cp.zeros([batch_size, out_height, out_width, channels])

        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride

                self.out[:, y, x, :] = cp.max(
                    X[:, y_stride:y_stride + self.pool_size, x_stride:x_stride + self.pool_size, :],
                    axis=(1, 2)
                )

        return self.out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        
        if self.result is None or not np.array_equal(self.result.shape, self.X.shape):
            self.result = cp.zeros(self.X.shape)
            
            
        batch_inds, channels_inds = np.repeat(range(batch_size), channels), np.tile(range(channels), batch_size)
        
        
        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride

                slice_X = self.X[:, y_stride: y_stride + self.pool_size, x_stride: x_stride + self.pool_size, :]
                slice_x_reshape = slice_X.reshape((batch_size, self.pool_size * self.pool_size, channels))

                maxpool_inds = cp.argmax(slice_x_reshape, axis=1).flatten()

                if self.tmp_result is None or not np.array_equal(self.tmp_result.shape, slice_x_reshape.shape):
                    self.tmp_result = cp.zeros(slice_x_reshape.shape)
 
                self.tmp_result[batch_inds, maxpool_inds, channels_inds] = d_out[batch_inds, y, x, channels_inds]

                self.result[:, y_stride: y_stride + self.pool_size, x_stride: x_stride + self.pool_size, :] = self.tmp_result.reshape(
                    (batch_size, self.pool_size, self.pool_size, channels)
                )

        return self.result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None
        self.name = 'flattener'

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = batch_size, height, width, channels
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}

    

class TanhLayer:
    def __init__(self):
        self.name = 'tanh'

    def forward(self, X):
        return cp.tanh(X)
        
    def backward(self, d_out):
        return 1-d_out**2

    def params(self):
        return {}


class SigmoidLayer:
    def __init__(self):
        self.name = 'sigmoid'

    def forward(self, X):
        return  1 / (1 + cp.exp(-X))

    def backward(self, d_out):
        return d_out * (1-d_out)

    def params(self):
        return {}


class RNN:
    def __init__(self, input_size, #output_size,
                 hidden_size):
        self.name = "RNN"
        self.input_size = input_size
        # self.output_size = output_size
        self.hidden_size = hidden_size
        self.W_ax = Param(0.001 * cp.random.randn(input_size, hidden_size))
        self.W_aa = Param(0.001 * cp.random.randn(hidden_size, hidden_size))
        self.B = Param(0.001 * cp.random.randn(1, hidden_size))
        # self.tanh = TanhLayer()
        

        
    def forward(self, input_X):
        batch_size = input_X.shape[0]
        self.input_X = cp.swapaxes(input_X, 0, 1)
 
        self.relus = [ReLULayer() for x in input_X]
        hidden = cp.zeros((batch_size, self.hidden_size))

        self.hidden_list = [hidden]
        self.y_preds = []

        for input_x_t, relu  in zip(self.input_X, self.relus):
            input_tanh = cp.dot(input_x_t, self.W_ax.value) + cp.dot(hidden, self.W_aa.value) + self.B.value
            
            hidden = relu.forward(input_tanh)
            
            if cp.any(cp.isnan(hidden)):
                return None
                
            self.hidden_list.append(hidden)

            # input_softmax = np.dot(self.Wya, hidden) + self.by
            # y_pred = self.softmax.forward(input_softmax)
            # self.y_preds.append(y_pred)

        return hidden

    def backward(self, d_out):
        
        for input_x_t, hidden, relu in reversed(list(zip(self.input_X, self.hidden_list[:-1], self.relus))):
            dtanh = relu.backward(d_out)
            self.B.grad += cp.array([cp.sum(d_out, axis=0)])
            self.W_ax.grad += cp.dot(input_x_t.T, dtanh)
            self.W_aa.grad += cp.dot(hidden.T, dtanh)
            d_out = cp.dot( dtanh, self.W_aa.value.T)
            
        return None
            
        
        
    def params(self):
        return {'W_ax': self.W_ax, 'W_aa': self.W_aa,  'B': self.B}


class LSTM:
    def __init__(self, input_size,  # output_size,
                 hidden_size):
        self.name = "LSTM"
        self.input_size = input_size
        # self.output_size = output_size
        self.hidden_size = hidden_size
        self.W_f = Param(0.001 * cp.random.randn(input_size+hidden_size, hidden_size))
        self.W_i = Param(0.001 * cp.random.randn(input_size+hidden_size, hidden_size))
        self.W_c = Param(0.001 * cp.random.randn(input_size+hidden_size, hidden_size))
        self.W_o = Param(0.001 * cp.random.randn(input_size+hidden_size, hidden_size))
        
        self.B_f = Param(0.001 * cp.random.randn(1, hidden_size))
        self.B_i = Param(0.001 * cp.random.randn(1, hidden_size))
        self.B_c = Param(0.001 * cp.random.randn(1, hidden_size))
        self.B_o = Param(0.001 * cp.random.randn(1, hidden_size))

        self.tanh = TanhLayer()
        self.sigmoid = SigmoidLayer()
        

    def forward(self, input_X):
        batch_size = input_X.shape[0]
        self.input_X = cp.swapaxes(input_X, 0, 1)

        hidden = cp.zeros((batch_size, self.hidden_size))
        cell_state = cp.zeros((batch_size, self.hidden_size))

        self.hidden_list = [hidden]
        self.cell_state_list = [cell_state]
        
        self.y_preds = []
        self.o_t_list = []
        self.i_t_list = []
        self.f_t_list = []
        self.C_t_wave_list = []

        for input_x_t in self.input_X:
            h_x_concat = cp.concatenate((hidden, input_x_t), axis=1)
            
            f_t = self.sigmoid.forward(h_x_concat @ self.W_f.value + self.B_f.value)
            i_t = self.sigmoid.forward(h_x_concat @ self.W_i.value + self.B_i.value)

            C_t_wave = self.tanh.forward(h_x_concat @ self.W_c.value + self.B_c.value)
            cell_state = f_t * cell_state + i_t*C_t_wave

            o_t = self.sigmoid.forward(h_x_concat @ self.W_o.value + self.B_o.value)
            hidden = o_t*self.tanh.forward(cell_state)


            self.hidden_list.append(hidden)
            self.cell_state_list.append(cell_state)
            self.o_t_list.append(o_t)
            self.i_t_list.append(i_t)
            self.f_t_list.append(f_t)
            self.C_t_wave_list.append(C_t_wave)

        return hidden #cp.swapaxes(hidden, 0, 1)


    def backward(self, d_out):
        # d_out = cp.swapaxes(d_out, 0, 1)

        d_cell_state = cp.zeros_like(self.cell_state_list[0])
        for input_x_t, hidden, cell_state, o_t, i_t, f_t, C_t_wave, prev_cell_state in reversed(list(zip(self.input_X, self.hidden_list[:-1], \
            self.cell_state_list, self.o_t_list, self.i_t_list, self.f_t_list, self.C_t_wave_list, self.cell_state_list[:-1]))):

            d_o_t = self.tanh.forward(cell_state) *  d_out
            d_C_t = d_cell_state + d_out * o_t * (1-self.tanh.forward(cell_state)**2)
            d_C_t_wave = d_C_t * i_t
            d_i_t = d_C_t * C_t_wave
            d_f_t = d_C_t * prev_cell_state

            d_f_t = f_t * (1-f_t) * d_f_t
            d_i_t = i_t * (1-i_t) * d_i_t
            # d_C_t_prev_wave = (1-C_t_wave**2) * d_C_t_wave

            d_o_t = o_t * (1-o_t) * d_o_t
            # print(self.W_f.value.T.shape, d_f_t.shape)
            # d_z_t = self.W_f.value.T @ d_f_t + self.W_i.value.T @ d_i_t + self.W_c.value.T @ d_C_t_wave + self.W_o.value.T @ d_o_t
            d_z_t = self.W_f.value @ d_f_t.T + self.W_i.value @ d_i_t.T + self.W_c.value @ d_C_t_wave.T + self.W_o.value @ d_o_t.T

            d_out =  d_z_t.T[:d_out.shape[0], :d_out.shape[1]]

            z = cp.concatenate((hidden, input_x_t), axis=1)

            self.W_f.grad += z.T @ d_f_t 
            self.B_f.grad += d_f_t.sum(axis=0).reshape(1, -1)

            self.W_i.grad += z.T @ d_i_t 
            self.B_i.grad += d_i_t.sum(axis=0).reshape(1, -1)

            self.W_c.grad += z.T @ d_C_t_wave 
            self.B_c.grad += d_C_t_wave.sum(axis=0).reshape(1, -1)

            self.W_o.grad += z.T @ d_o_t 
            self.B_o.grad += d_o_t.sum(axis=0).reshape(1, -1)
        return None

    def params(self):
        return {'W_f': self.W_f, 'W_i': self.W_i, 'W_c': self.W_c, 'W_o': self.W_o, 
            'B_f': self.B_f, 'B_i': self.B_i, 'B_c': self.B_c, 'B_o': self.B_o }