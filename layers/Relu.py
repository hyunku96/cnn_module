import numpy as np

class ReLU(object):

    def forward(self, x):
        self.x = x
        return np.maximum(x,0)

    def backward(self, dEd):
        self.dEd = dEd
        self.dEd[self.x <0] =0
        return self.dEd