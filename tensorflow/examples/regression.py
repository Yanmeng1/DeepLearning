import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# sample
x_data = np.linspace(-0.5,0.5,200).reshape(200,1)   #[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

# placeholder
x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

# define neural network - middle
weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
# [1,10] = num * [1,10] + [1,10]
wx_plus_b_L1 = tf.matmul(x,weights_L1) + biases_L1
a1 = tf.nn.tanh(wx_plus_b_L1)

# define neural network - output
weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
# [1,1] = [1,10] * [10,1] + [1,1]
wx_plus_b_L2 = tf.matmul(a1, weights_L2) + biases_L2
a2 = tf.nn.tanh(wx_plus_b_L2)

prediction = a2

# cost function
loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(20000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()