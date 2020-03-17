import numpy as np

class Softmax(object):
    def predict(self, prediction, label):
        self.label = label
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        prediction -= np.max(prediction)
        exp_prediction = np.exp(prediction)
        self.softmax = exp_prediction / np.sum(exp_prediction)
        return self.softmax


    def backward(self):
        gradient = self.softmax.copy()
        gradient[0][self.label] -=1
        return gradient




