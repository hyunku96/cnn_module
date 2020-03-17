import numpy as np
from functools import reduce
import math

class FullyConnect(object):
    def __init__(self, shape, output_num=10):
        self.input_shape = tuple(map(int, shape))
        self.output_shape = [output_num]
        self.output_channel = output_num
        input_len = reduce(lambda x,y:x*y, shape[:]) #shape 이 몇차원일지 모를때 사용

        self.weights = np.random.standard_normal((int(input_len), output_num))
        self.bias = np.random.standard_normal(output_num)
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = np.reshape(x.reshape(-1), (1, -1))
        output = np.dot(self.x, self.weights)
        for b in range(self.output_channel):
            output[0][b] += self.bias[b]
        return output

    def backward(self, gradient):
        self.w_gradient = np.dot(self.x.T, np.reshape(gradient, (1, 10)))
        self.b_gradient = gradient
        next_gradient = np.dot(gradient, self.weights.T)
        next_gradient = np.reshape(next_gradient, self.input_shape)
        return next_gradient

    def update(self, learning_rate = 0.01):
        self.weights -= self.w_gradient * learning_rate
        self.bias -=learning_rate * np.reshape(self.b_gradient, (-1))
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
