import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, *args, **kwargs):
        '''
        args[0] : weight
        args[1] : bias
        '''
        return self.update(args[0], args[1])

    def update(self, w_gradient, b_gradient):
        w_gradient *= self.lr
        b_gradient *= self.lr
        return w_gradient, b_gradient