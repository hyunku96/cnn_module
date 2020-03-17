import numpy as np

class Conv(object):
    def __init__(self, shape, output_channels, ksize, stride, padding_size):
        self.input_shape = tuple(map(int, shape)) #height
        self.input_channels = shape[0]

        self.output_channels = output_channels
        self.ksize = ksize
        self.stride = stride
        self.matrix_input = []
        self.padding_size = padding_size

        weights_scale = np.sqrt(shape[0] * shape[1] * shape[2])
        self.weights = np.random.standard_normal((self.output_channels, self.input_channels,ksize, ksize))/weights_scale *np.sqrt(2)
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        self.matrix_weights = np.zeros((self.input_channels * self.ksize * self.ksize, self.output_channels))


        self.gradient = np.zeros(( self.output_channels, int((shape[1] + 2*padding_size - ksize)/stride+1), (int)((shape[1] + 2*padding_size - ksize)/stride+1)))
        self.w_graidient = np.zeros(self.weights.shape)
        self.bias_graidnet = np.zeros(self.bias.shape)

        self.output_shape = self.dEd.shape


        if (self.input_shape[1] +2 * padding_size - ksize) % stride != 0:
            print('input tensor width can\'t fit stride')

    def forward(self, input):
        self.matrix_weights = np.reshape(self.weights, (self.output_channels, -1)).T
        npad = ((0, 0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size))
        self.padding_input = np.pad(input, npad,'constant', constant_values=0.0)



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

        conv_out=[]
        conv_out = np.dot(self.matrix_input, self.matrix_weights).T
        conv_out = np.reshape(conv_out,  self.gradient.shape)
        for c in range(self.output_channels):
            conv_out[c] += self.bias[c]

        return conv_out

    #nn 에서 backward랑 같은 기능. 다음 gradient를 위한 계산
    def backward(self, gradient):
        self.gradient = gradient
        conv_gradient = np.reshape(self.gradient, (self.output_channels, -1)).T
        self.w_graidient = np.reshape(np.dot(self.matrix_input.T, conv_gradient).T, (-1))
        self.w_graidient =  np.reshape(self.w_graidient, self.weights.shape)

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
    def update(self, learning_rate = 0.01):
        self.weights -= self.w_graidient * learning_rate
        self.bias -= self.bias * learning_rate

        self.w_graidient = np.zeros(self.weights.shape)
        self.bias_graidnet = np.zeros(self.bias.shape)
