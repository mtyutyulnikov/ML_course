import numpy as np

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
    loss = reg_strength * np.sum(W ** 2)
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
        s = np.max(predictions, axis=1)
        e_x = np.exp(predictions - s[:, np.newaxis])
        div = np.sum(e_x, axis=1)
        return e_x / div[:, np.newaxis]
    else:
        exps = np.exp(predictions - np.max(predictions))
        return exps / np.sum(exps)


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
        target_index = np.asarray([target_index])
    if len(target_index.shape) == 1:
        probs = probs[target_index]
    else:
        probs = probs[range(probs.shape[0]), target_index[:, 0]]
        
    probs = np.clip(probs, 0.001, 1)

    return -np.mean(np.log(probs))


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
        index = np.asarray([target_index])
    else:
        index = target_index.copy()

    if index.ndim == 1 and index.size > 1:
        index = index.reshape(-1, 1)

    prob = softmax(preds.copy())
    if np.isnan(prob).sum() > 0:
        print(prob.shape, 'prob')
        
    loss = cross_entropy_loss(prob, index)

    y = np.zeros_like(preds)

    if len(index.shape) == 1:
        y[index] = 1
    else:
        y[range(y.shape[0]), index[:, 0]] = 1

    
    dprediction = prob - 1 * y
    if np.isnan(dprediction).sum() > 0:
        print(dprediction.shape, 'grad1')
        
    if preds.ndim == 2:
        dprediction = dprediction / preds.shape[0]
    
    if np.isnan(dprediction).sum() > 0:
        print(dprediction.shape, 'grad2')
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.name = 'relu'
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        # print(X)
        
        self.indexes = X < 0
        result = X.copy()
        result[self.indexes] = 0
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
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")

        d_result = d_out.copy()
        d_result[self.indexes] = 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.name = 'FC'

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        self.X = X
        
        res = np.dot(X, self.W.value) + self.B.value
        
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
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        # raise Exception("Not implemented!")

        self.W.grad = np.dot(self.X.T, d_out)  # dL/dW = dL/dZ * dZ/dW = X.T * dL/dZ
        self.B.grad = np.array([np.sum(d_out, axis=0)])  # dL/dB = dL/dZ * dZ/dB = I * dL/dZ
        gradX = np.dot(d_out, self.W.value.T)  # dL/dX = dL/dZ * dZ/dX = dL/dZ * W.T
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
            np.random.randn(filter_size, filter_size,
                            in_channels, 
                            out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = stride
        self.name = 'conv'

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = 0
        out_width = 0
        self.X = X.copy()

        out_height = int(np.floor(height + 2 * self.padding - self.filter_size)/self.stride + 1)
        out_width = int(np.floor(width + 2 * self.padding - self.filter_size)/self.stride + 1)

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        out = np.zeros([batch_size, out_height, out_width, self.out_channels])

        self.X = np.pad(
            array=self.X,
            pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='constant'
        )

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        slice_W = self.W.value.reshape(-1, self.out_channels)
        if np.isnan(slice_W).sum() > 0:
            print(slice_W.shape, 'slice_W conv')
            
        # print(self.W.value.shape, slice_W.shape)
        for y in range(0, out_height):
            for x in range(0, out_width):
                # TODO: Implement forward pass for specific location
                slice_X = self.X[:, y*self.stride:y*self.stride + self.filter_size, \
                                       x*self.stride:x*self.stride + self.filter_size, :] \
                    .reshape(batch_size, -1)

                # out[:, y, x, :] = slice_X.dot(slice_W) + self.B.value
                out[:, y, x, :] = slice_X.dot(slice_W) + self.B.value
                
                if np.isinf(slice_X).sum()>0:
                    print('slice_x', slice_X.shape)
                if np.isinf(slice_W).sum()>0:
                    print('slice_w', slice_W.shape)
                
                
                
                # if np.isinf(slice_X.dot(slice_W)).sum()>0:
                #     print(np.isnan(slice_X.dot(slice_W)).sum())
                #     print('dot', slice_X.dot(slice_W).shape)
                #     print('sl_w', slice_W.shape)
                #     print('sl_x', slice_X.shape)

                if np.isnan(self.B.value).sum()>0:
                    print('B', B.shape)
                
#                 slice_X = self.X.value[:, y*self.stride:y*self.stride + self.filter_size, \
#                                        x*self.stride:x*self.stride + self.filter_size, :] 
# #                     .reshape(batch_size, -1)

#                 print(slice_X.shape, self.W.value[np.newaxis, :, :, :].shape)
#                 print((slice_X * self.W.value[np.newaxis, :, :, :]).shape)
                      
                      
#                 out[:, y, x, :] = np.sum(
#                     slice_X * self.W.value[np.newaxis, :, :, :],
#                     axis=(1, 2)
#                 )
#                 # print(out.shape)
                pass
        
        # raise Exception("Not implemented!")
        if np.isnan(out).sum() > 0:
            print(out.shape, 'out conv')
        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        slice_W = self.W.value.reshape(-1, self.out_channels)
        grad_X = np.zeros_like(self.X)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                slice_X = self.X[:, y*self.stride:y*self.stride + self.filter_size, x*self.stride:x*self.stride + self.filter_size, :] \
                    .reshape(batch_size, -1)

                self.W.grad += np.dot(slice_X.T, d_out[:, y, x, :]).reshape((self.filter_size, self.filter_size,
                                                                             self.in_channels, out_channels))

                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)

                grad_X[:, y*self.stride:y*self.stride + self.filter_size, x*self.stride:x*self.stride + self.filter_size, :] += np.dot(d_out[:, y, x, :], slice_W.T) \
                    .reshape((batch_size, self.filter_size, self.filter_size, self.in_channels))

                pass
        return grad_X[:, self.padding:height - self.padding, self.padding:width - self.padding, :]
        # raise Exception("Not implemented!")

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
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

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros([batch_size, out_height, out_width, channels])

        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride

                out[:, y, x, :] = np.max(
                    X[:, y_stride:y_stride + self.pool_size, x_stride:x_stride + self.pool_size, :],
                    axis=(1, 2)
                )

        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        result = np.zeros(self.X.shape)
        batch_inds, channels_inds = np.repeat(range(batch_size), channels), np.tile(range(channels), batch_size)

        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride

                slice_X = self.X[:, y_stride: y_stride + self.pool_size, x_stride: x_stride + self.pool_size, :]
                slice_x_reshape = slice_X.reshape((batch_size, self.pool_size * self.pool_size, channels))

                maxpool_inds = np.argmax(slice_x_reshape, axis=1).flatten()

                tmp_result = np.zeros(slice_x_reshape.shape)
                tmp_result[batch_inds, maxpool_inds, channels_inds] = d_out[batch_inds, y, x, channels_inds]

                result[:, y_stride: y_stride + self.pool_size, x_stride: x_stride + self.pool_size, :] = tmp_result.reshape(
                    (batch_size, self.pool_size, self.pool_size, channels)
                )

        return result

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

    
    
class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.name = 'dropout'

    def forward(self, X):
        self.mask = np.random.binomial(1,self.dropout_rate,size=X.shape) / self.dropout_rate
        out = X * self.mask
        return out

    def backward(self, d_out):
        return d_out * self.mask

    def params(self):
        # No params!
        return {}