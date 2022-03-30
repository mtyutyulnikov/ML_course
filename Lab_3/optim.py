import numpy as np


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
        self.epsilon = epsilon
        
        self.t = 1
        self.v = 0
        self.s = 0
        
        
    def update(self, weights, grad, learning_rate):
        self.v = self.beta_1 * self.v + (1 - self.beta_1) * grad
        self.s = self.beta_2 * self.s + (1 - self.beta_2) * np.square(grad)

        v_bias_corr = self.v / (1 - self.beta_1 ** self.t)
        s_bias_corr = self.s / (1 - self.beta_2 ** self.t)
        weights -= learning_rate * v_bias_corr / (np.sqrt(s_bias_corr)+ self.epsilon)
        self.t+=1
        
        return weights
        

