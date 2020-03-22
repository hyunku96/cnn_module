import numpy as np
from . import Dcg

class MSE:
    '''
    Mean Squared Error loss function
    compute MSE and initiate back propagation
    '''
    def __init__(self, output, label, optimizer):
        self.output = output
        self.label = label
        self.optim = optimizer
        self.dcg = Dcg.DCG.getDCG()

    def backward(self):
        if self.dcg.len() <= 0:
            print("Cannot find computation to calculate gradient")
            return
        else:
            loss = self.output - self.label
            tmp = self.dcg.pop()
            gradient = tmp.function(tmp.data, loss, self.optim)
            while self.dcg.len() > 0:
                tmp = self.dcg.pop()
                gradient = tmp.function(tmp.data, gradient, self.optim)