import numpy as np
from tqdm import *
from layers.Conv import conv
from layers.FullyConnect import fullyconnect
from layers.MaxPooling import maxpooling
from layers.Sigmoid import sigmoid
from layers.Relu import relu
from layers.MSE import MSE
from optimizers.SGD import SGD
from layers.Dcg import zero_grad
from utils.LoadData import get_MNIST

# build model
class net:
    def __init__(self):
        self.conv1 = conv(1, 4, 3)
        self.conv2 = conv(4, 8, 3)
        self.fc1 = fullyconnect(7*7*8, 7*7)
        self.fc2 = fullyconnect(49, 10)
        self.maxpooling = maxpooling(2, 2)
        self.relu = relu()
        self.sigmoid = sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpooling(x)
        x = self.relu(self.conv2(x))
        x = self.maxpooling(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Data Loading
TrainImg, TrainLabel, TestImg, TestLabel = get_MNIST()

# train model
model = net()
optimizer = SGD(lr=0.01)
for epoch in range(100):
    zero_grad()
    indexes = np.random.permutation(len(TrainLabel))
    for index in tqdm(indexes):
        img = TrainImg[index]
        output = model.forward(img)
        label = TrainLabel[index]
        loss = MSE(output, label, optimizer)
        loss.backward()

    # test model
    acc = 0
    for i in range(len(TestLabel)):
        output = model.forward(TestImg[i])
        if output.argmax() == TestLabel[i].argmax():
            acc += 1
    print("epoch:{0}, accuracy:{1}".format(epoch, acc/len(TestLabel)))