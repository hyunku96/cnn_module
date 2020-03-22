import numpy as np

class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, epsilon=1e-7):
        '''
        Adam optimizer
        :param lr: learning rate
        :param b1: momentum factor
        :param b2: RMSprop factor
        :param epsilon: constant to avoid zero divide
        '''
        self.lr = lr
        self.t = self.m_weight = self.m_bias = self.v_weight = self.v_bias = 0
        self.b1, self.b2, self.epsilon = b1, b2, epsilon

    def __call__(self, *args, **kwargs):
        '''
        args[0] : weight
        args[1] : bias
        '''
        return self.update(args[0], args[1])

    def update(self, w_gradient, b_gradient):
        self.t += 1
        self.m_weight = self.b1 * self.m_weight + (1 - self.b1) * w_gradient
        self.m_bias = self.b1 * self.m_bias + (1 - self.b1) * b_gradient
        self.v_weight = self.b2 * self.v_weight + (1 - self.b2) * (w_gradient**2)
        self.v_bias = self.b2 * self.v_bias + (1 - self.b2) * (b_gradient**2)
        m_hat_weight = self.m_weight / (1 - self.b1**self.t)
        m_hat_bias = self.m_bias / (1 - self.b1**self.t)
        v_hat_weight = self.v_weight / (1 - self.b2**self.t)
        v_hat_bias = self.v_bias / (1 - self.b2**self.t)
        w_gradient -= self.lr * m_hat_weight / (np.sqrt(v_hat_weight) + self.epsilon)
        b_gradient -= self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
        return w_gradient, b_gradient