# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# sample
x_data = np.linspace(-0.5,0.5,200).reshape(200,1)   #[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 200, 1, 200, 200

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

plt.figure()
plt.scatter(x_data, y_data)

y_pred = []
for t in range(50000):
    # Forward pass: compute predicted y
    # [N, D_in] * [D_in, H] = [N, H]
    x = x_data
    y = y_data

    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    # [N, H] * [H, D_out] = [N, out]
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()

    if t%2000 == 1999:
        print(t + 1, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)  # [H, N] * [N,out] = [H, out]
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.plot(x_data, y_pred, 'r-', lw=1)
plt.show()