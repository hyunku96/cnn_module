import numpy as np
from . import Dcg


class maxpooling:
    '''
    def __init__(self, hsize, wsize):
        self.hsize, self.wsize = hsize, wsize
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):

        #args[0] : input data

        if args[0].shape[1] % self.hsize != 0:
            print('maxpooling size height is not fit conv after')
        if args[0].shape[2] % self.wsize != 0:
            print('maxpooling size width is not fit conv after')
        return self.forward(args[0])

    def forward(self, x):
        out = np.zeros((x.shape[0], int(x.shape[1] / self.hsize), int(x.shape[2] / self.wsize)))
        index = np.zeros(x.shape)

        for c in range(x.shape[0]):
            for h in range(0, x.shape[1], self.hsize):
                for w in range(0, x.shape[2], self.wsize):
                    out[c, int(h / self.hsize), int(w / self.wsize)] = np.max(x[c, h:h + self.hsize, w:w + self.wsize])
                    maxloc = np.argmax(x[c, h:h + self.hsize, w:w + self.wsize])
                    index[c, h + int(maxloc / self.hsize), w + maxloc % self.wsize] = 1

        tmp = Dcg.node(index)
        tmp.function = self.backward
        self.dcg.append(tmp)
        return out

    def backward(self, input, gradient):
        gradient = np.reshape(gradient, input.shape)
        return np.repeat(np.repeat(gradient, self.hsize, axis=1), self.wsize, axis=2) * input
    '''
    def __init__(self, hsize, wsize):
        self.h_size, self.w_size = hsize, wsize
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        '''
        args[0] : input data
        '''
        return self.forward(args[0])

    def forward(self, input):
        '''
        slicing every pooling area in input then find max value.
        Assemble max values and reshape it.
        If feature map divided by pooling size is not integer, this method will not deal with input's margin.
        '''
        tmp = Dcg.node(input)
        tmp.function = self.backward
        self.dcg.append(tmp)

        result = []
        for depth in range(input.shape[0]):
            out = []
            for height in range(0, input.shape[1] - 1, self.h_size):
                for width in range(0, input.shape[2] - 1, self.w_size):
                    out.append(np.max(input[depth, height:height + self.h_size, width:width + self.w_size]))
            result.append(np.reshape(out, (int(input.shape[1] / self.h_size), int(input.shape[2] / self.w_size))))
        return result

    def backward(self, input, gradient):
        gradient = np.reshape(gradient,
                              (input.shape[0], int(input.shape[1] / self.h_size), int(input.shape[2] / self.w_size)))
        mask = np.zeros((input.shape[0], input.shape[1], input.shape[2]))
        for depth in range(input.shape[0]):
            for height in range(0, input.shape[1] - 1, self.h_size):
                for width in range(0, input.shape[2] - 1, self.w_size):
                    area = input[depth][height:height + self.h_size, width:width + self.w_size]
                    if area.any():  # if area's elements were all zero(caused by relu), gradients shall not pass
                        maxloc = area.argmax()
                        # assign rear layer's gradient to maxpooling & ReLU layer's gradient
                        mask[depth, height + int(maxloc / self.h_size), width + maxloc % self.w_size] = gradient[
                            depth, int(height / self.h_size), int(width / self.w_size)]

        return mask