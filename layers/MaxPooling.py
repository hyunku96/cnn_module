import numpy as np
import matplotlib.pyplot as plt


class MaxPooling(object):
    def __init__(self, shape, size):
        self.input_shape = shape
        self.size = size
        self.stride = size
        self.output_channel = shape[0]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0], shape[1] / self.stride, shape[2] / self.stride]

        if shape[1] % self.stride != 0:
            print('maxpooling size height is not fit conv after')
        if shape[2] % self.stride != 0:
            print('maxpooling size width is not fit conv after')

    def forward(self, x):
        out = np.zeros((x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride)))

        for c in range(x.shape[0]):
            for h in range(0, x.shape[1], self.stride):
                for w in range(0, x.shape[2], self.stride):
                    out[c, int(h / self.stride), int(w / self.stride)] = np.max(x[c, h:h + self.size, w:w + self.size])
                    index = np.argmax(x[c, h:h + self.size, w:w + self.size])
                    self.index[c, h + int(index / self.stride), w + index % self.stride] = 1
        return out

    def backward(self, gradient):
        return np.repeat(np.repeat(gradient, self.stride, axis=1), self.stride, axis=2) * self.index

