import numpy as np
from tqdm import *
from layers.FullyConnect import fullyconnect
from layers.Sigmoid import sigmoid
from layers.Tanh import tanh
from layers.Leaky_relu import leaky_relu
from layers.Dropout import dropout
from layers.BCE import BCE
from optimizers.SGD import SGD
from layers.Dcg import zero_grad
from utils.LoadData import get_MNIST
import matplotlib.pyplot as plt


# build model
class Generator:
    def __init__(self):
        self.fc1 = fullyconnect(100, 256)
        self.fc2 = fullyconnect(256, 512)
        self.fc3 = fullyconnect(512, 1024)
        self.fc4 = fullyconnect(1024, 28*28)
        self.leaky_relu = leaky_relu(0.2)
        self.tanh = tanh()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x


class Discriminator:
    def __init__(self):
        self.fc1 = fullyconnect(28*28, 1024)
        self.fc2 = fullyconnect(1024, 256)
        self.fc3 = fullyconnect(256, 64)
        self.fc4 = fullyconnect(64, 1)
        self.leaky_relu = leaky_relu(0.2)
        self.dropout = dropout(0.3)
        self.sigmoid = sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x


# Data Loading
TrainImg, TrainLabel, TestImg, TestLabel = get_MNIST()

G = Generator()
D = Discriminator()
G_optimizer = SGD(lr=0.01)
D_optimizer = SGD(lr=0.01)

def G_update():
    G.fc1.update()
    G.fc2.update()
    G.fc3.update()
    G.fc4.update()

def D_update():
    D.fc1.update()
    D.fc2.update()
    D.fc3.update()
    D.fc4.update()


for epoch in range(200):
    indexes = np.random.permutation(len(TrainLabel))
    for index in tqdm(indexes):
        # train discriminator
        zero_grad()
        # train real
        D_output = D.forward(TrainImg[index])
        D_real_loss = BCE(D_output, 1, D_optimizer)
        D_real_loss.backward()
        D_update()

        # train fake
        z = np.random.rand(100)
        D_output = D.forward(G.forward(z))
        D_fake_loss = BCE(D_output, 0, D_optimizer)
        D_real_loss.backward()
        D_update()

        # train generator
        zero_grad()
        z = np.random.rand(100)

        G_output = G.forward(z)
        D_output = D.forward(G_output)
        G_loss = BCE(D_output, 1, G_optimizer)
        G_loss.backward()
        G_update()

    z = np.random.rand(100)
    G_output = G.forward(z)
    D_output = D.forward(G_output)
    G_output = np.reshape(G_output, (28, 28))
    print("epoch:{0}, D_output:{1}".format(epoch, D_output))
    plt.imsave("{0}epoch.png".format(epoch), G_output)
