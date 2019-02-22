# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Building a neural network without framework
class network():

    def __init__(self, layers, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weights = []
        self.bias = []
        self.weights.append(0)
        self.bias.append(0)
        for i in range(1, len(layers)):
            np.random.seed(9 + i)
            self.weights.append(np.random.rand(layers[i-1], layers[i]))
            self.bias.append(0)

    def train(self, X, Y):
        # Forward
        z = [0] * len(self.weights)
        a = [0] * len(self.weights)

        a[0] = X
        for i in range(1, len(self.weights)):
            z[i] = np.dot(a[i-1], self.weights[i]) + self.bias[i]
            a[i] = np.tanh(z[i])
        Y_prediction = a[-1]
        loss = np.square(Y - Y_prediction).sum()

        # Backpropagation
        dz = [0] * len(z)
        da = [0] * len(a)
        dw = [0] * len(self.weights)
        db = [0] * len(self.bias)

        da[len(a) - 1] = 2.0 * (Y_prediction - Y)
        for i in range(len(self.weights) - 1, 0, -1):
            dz[i] = da[i] * (1 - np.square(a[i]))
            dw[i] = (1.0 / self.batch_size) * np.dot(a[i-1].T, dz[i])
            self.weights[i] -= self.learning_rate * dw[i]
            db[i] = (1.0 / self.batch_size) * dz[i].sum(0).reshape(1,-1)
            self.bias[i] -= self.learning_rate * db[i]
            da[i-1] = np.dot(dz[i], self.weights[i].T)
        return loss

    def prediction(self, X):
        z = [0] * len(self.weights)
        a = [0] * len(self.weights)
        a[0] = X
        for i in range(1, len(self.weights)):
            z[i] = np.dot(a[i - 1], self.weights[i]) + self.bias[i]
            a[i] = np.tanh(z[i])
        return a[-1]

# sample
x_data = np.linspace(-0.5, 0.5, 200).reshape(200, 1)
np.random.seed(0)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = np.square(x_data) * x_data + noise

net = network(layers=[1, 20, 20, 1], batch_size = 200, learning_rate = 0.01)

count = 0
loss = []
for i in range(5000):
    l = net.train(x_data, y_data)
    if i % 99 == 0:
        count += 1
        loss.append(l)
        print('iter ', i ,'loss ', l)

plt.figure()
x_test = np.linspace(-0.5, 0.5, 200).reshape(200, 1)
y_test = net.prediction(x_test)
# plt.plot(range(count), loss, 'r-', lw=1)
plt.scatter(x_data, y_data)
plt.plot(x_test, y_test, 'r-', lw=4)
plt.show()

