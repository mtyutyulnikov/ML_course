import numpy as np
import cupy as cp
from time import time

class SGD:
    """
    Implements vanilla SGD update
    """
    def update(self, w, d_w, learning_rate):
        """
        Performs SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        return w - d_w * learning_rate


class MomentumSGD:
    """
    Implements Momentum SGD update
    """
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = 0

    def update(self, w, d_w, learning_rate):
        """
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        # TODO Implement momentum update
        # Hint: you'll need to introduce some variables to remember
        # velocity from the previous updates
        # raise Exception("Not implemented!")

        self.velocity = self.momentum * self.velocity - learning_rate * d_w
        w = w + self.velocity
        return w

    
class Adam():
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon = 1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = cp.array(epsilon).astype(np.float32)
        
        self.t = 1
        self.v = 0
        self.s = 0
        
        self.sqrt_res = None
        
        
    
    def update(self, weights, grad, learning_rate):
        
        kernel_update = cp.RawKernel(r'''
             extern "C" __global__
             void adam_update(float* weights, const int N, const float* learning_rate, 
                                 const float* v_bias_corr, const float* s_bias_corr,  const float* epsilon) {
                 int tid = blockDim.x * blockIdx.x + threadIdx.x;
                
                 if (tid < N){
                     weights[tid] -= learning_rate[0]*v_bias_corr[tid] / (sqrt(s_bias_corr[tid]) + epsilon[0]);
                 }
             }
             ''', 'adam_update')
        
        
        self.v = self.beta_1 * self.v + (1 - self.beta_1) * grad
        self.s = self.beta_2 * self.s + (1 - self.beta_2) * cp.square(grad)
        
        v_bias_corr = self.v / (1 - self.beta_1 ** self.t)
        s_bias_corr = self.s / (1 - self.beta_2 ** self.t)
        
        
        # print('LEN W', len(weights.flatten()))        
        
        learning_rate = cp.array(learning_rate).astype(np.float32)
        
        kernel_update((len(weights.flatten())//1024+1,),(1024,), \
                      (weights, len(weights.flatten()), learning_rate, v_bias_corr, s_bias_corr, self.epsilon))
        
        
        # w -= learning_rate * v_bias_corr / (cp.sqrt(s_bias_corr)+ self.epsilon)
        self.t+=1
        
        

