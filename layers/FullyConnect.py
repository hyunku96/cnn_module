import numpy as np
from . import Dcg

class fullyconnect:
    '''
    fully-connected layer
    '''
    def __init__(self, input_node, output_node):
        '''
        get instance of DCG and init pramters
        '''
        self.w = np.random.rand(input_node, output_node) - 0.5
        self.b = np.random.rand(output_node) - 0.5
        self.dw, self.db = None, None
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        '''
        args[0] is 2 dimension when model is batch-mode
        '''
        return self.forward(args)

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        data = np.reshape(data, (1, -1))
        tmp = Dcg.node(data)
        tmp.function = self.backward
        self.dcg.append(tmp)
        output = np.dot(data, self.w) + self.b
        return output

    def backward(self, input, gradient, optimizer):
        dw = np.dot(np.reshape(input, (-1, 1)), gradient)
        db = gradient
        gradient = np.dot(gradient, self.w.T)
        self.dw, self.db = optimizer(dw, db)
        self.db = np.ravel(self.db, order='C')
        return gradient

    def update(self):
        self.w -= self.dw
        self.b -= self.db