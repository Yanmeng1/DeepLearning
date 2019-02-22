import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

## batch_size优化
batch_size = 100
batch_num = mnist.train.num_examples // batch_size

# examples = batch_size = 100
# [batch_size,784]
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.Variable(0.001, dtype=tf.float32)

###优化1.设置随机初始化
# 定义10个神经元
## 网络结构优化，初始值设定优化
# weight_L1 = tf.Variable(tf.zeros([784,10]))
# bias_L1 = tf.Variable(tf.zeros([1,10]))
# (u-2sigma, u+2sigma)之外截断

# [batch_size, 2000] = [batch_size, 784] * [784, 2000]
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([1, 500]) + 0.1)
Z1 = tf.matmul(x, W1) + b1
A1 = tf.tanh(Z1)
A1_dropout = tf.nn.dropout(A1, keep_prob)

# [batch_size, 2000] = [batch_size, 2000] * [2000, 2000]
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([1, 300]) + 0.1)
Z2 = tf.matmul(A1_dropout, W2) + b2
A2 = tf.tanh(Z2)
A2_dropout = tf.nn.dropout(A2, keep_prob)

# [batch_size, 1000] = [batch_size, 2000] * [2000, 1000]
W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([1, 10]) + 0.1)
Z3 = tf.matmul(A2_dropout, W3) + b3
# A3 = tf.tanh(Z3)
# A3_dropout = tf.nn.dropout(A3, keep_prob)

# [batch_size, 10] = [batch_size, 1000] * [1000, 10]
# W4 = tf.Variable(tf.truncated_normal([20,10],stddev=0.1))
# b4 = tf.Variable(tf.zeros([1,10]) + 0.1)
# Z4 = tf.matmul(A3_dropout, W4) + b4
prediction = tf.nn.softmax(Z3)

###优化1.如果输出神经元是线性的，那么二次代价函数就是一种合适的选择；如果输出神经元是s型函数，那么比较适合用交叉熵函数。
## 二次损失函数与激活函数的导数成正比
# loss = tf.reduce_mean(tf.square(y - prediction))
## 交叉熵损失函数能加速收敛：损失函数与（prediction - y）成正比
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Z3))

## 梯度下降优化，梯度为0.2
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 结果存放在boolean型的列表 tf.argmax()返回最大数值的下标，1返回每一行最大值下标
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    temp_y = []
    temp_x = []
    temp_z = []
    ## 训练次数改动 21 ？
    for epoch in range(21):
        sess.run(tf.assign(learning_rate, 0.001 * (0.9 ** epoch)))
        for batch in range(batch_num):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        lr = sess.run(learning_rate)
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(test_acc) + ", Training Accuracy " + str(
            train_acc) + ", lr " + str(lr))
        temp_x.append(epoch)
        temp_y.append(test_acc)
        temp_z.append(train_acc)
    plt.figure()
    plt.plot(temp_x, temp_y, 'r-', lw=1)
    plt.plot(temp_x, temp_z, 'g-', lw=1)
    plt.show()