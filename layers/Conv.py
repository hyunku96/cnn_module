import numpy as np
from . import Dcg

class conv(object):
    '''
    weights scale을 삭제함, conv(input_channel, knum, ksize)로 선언 할 수 있음
    '''
    def __init__(self, input_channels, knum, ksize, stride=1, padding_size=1):
        self.input_channels = input_channels
        self.knum = knum
        self.ksize = ksize
        self.erosion = int(self.ksize/2)
        self.stride = stride
        self.padding_size = padding_size
        self.weights, self.bias = None, None
        self.dw, self.db = None, None
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        '''
        args[0].shape[0, 1, 2] : depth, height, width of input
        args[0] is 4 dimension when model is batch-mode
        self.k : layer's kernel - 4 dimension(kernel num, input depth, kernel height, kernel width)
        self.b : layer's bias - 3 dimension(kernel num, output width, output height)
        '''
        if self.weights is None and self.bias is None:
            self.weights = np.random.rand(self.knum, self.input_channels, self.ksize, self.ksize) - 0.5
            self.bias = np.random.rand(self.knum, int((np.array(args[0]).shape[1] + 2*self.padding_size - self.ksize)/self.stride+1), int((np.array(args[0]).shape[2] + 2*self.padding_size - self.ksize)/self.stride+1)) - 0.5
        return self.forward(args[0])

    def forward(self, input):
        '''
        zero pad -> transfer img to matrix -> convolution(dot product) -> reshape matrix to tensor
        '''
        input = np.array(input)
        self.matrix_weights = np.reshape(self.weights, (self.knum, -1)).T
        npad = ((0, 0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size))
        self.padding_input = np.pad(input, npad,'constant', constant_values=0.0)
        tmp = Dcg.node(None)  # backward doesn't need input(self.matrix_input already exist)
        tmp.function = self.backward
        self.dcg.append(tmp)

        # im2col 함수
        matrix = np.zeros(self.padding_input.shape[0] * self.ksize *self.ksize)
        arr = []
        arr1 = []
        for h in range(0, self.padding_input.shape[1] - self.ksize+1, self.stride):
            for w in range(0, self.padding_input.shape[2] - self.ksize +1, self.stride):
                for c in range(self.padding_input.shape[0]):
                    arr1 = self.padding_input[c:c+1, h:h+self.ksize, w:w+self.ksize].copy()
                    arr1 = np.array(arr1).flatten()
                    arr = np.append(arr, arr1)

                matrix = np.vstack([matrix, arr])
                arr = []

        matrix = matrix[1:, :]
        self.matrix_input = np.array(matrix)

        conv_out = np.dot(self.matrix_input, self.matrix_weights).T
        conv_out = np.reshape(conv_out, (self.knum, int((input.shape[1] + 2*self.padding_size - self.ksize)/self.stride+1), (int)((input.shape[2] + 2*self.padding_size - self.ksize)/self.stride+1)))
        conv_out += self.bias

        return conv_out

    #nn 에서 backward랑 같은 기능. 다음 gradient를 위한 계산
    def backward(self, input, gradient, optimizer):
        self.gradient = gradient
        conv_gradient = np.reshape(self.gradient, (self.knum, -1)).T
        w_graidient = np.reshape(np.dot(self.matrix_input.T, conv_gradient).T, (-1))
        w_graidient = np.reshape(w_graidient, self.weights.shape)
        # update weights here
        self.dw, self.db = optimizer(w_graidient, gradient)

        padding_gradient = np.dot(conv_gradient, self.matrix_weights.T)
        next_gradient = np.zeros(self.padding_input.shape)

        index=0
        for h in range(0,next_gradient.shape[1] - self.ksize+1, self.stride):
            for w in range(0, next_gradient.shape[2] - self.ksize+1, self.stride):
                next_gradient[:, h:h + self.ksize, w:w + self.ksize] += np.reshape(padding_gradient[index], (self.input_channels, self.ksize, self.ksize))
                index+=1

        next_dEd = next_gradient[:, self.padding_size :next_gradient.shape[1] - self.padding_size, self.padding_size :next_gradient.shape[2] - self.padding_size]
        return next_dEd

    #Conv 의 weights 와 bias update
    def update(self):
        self.weights -= self.dw
        self.bias -= self.db
