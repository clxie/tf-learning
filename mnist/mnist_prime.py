"""Functions for downloading and reading MNIST data. START"""
""" 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# pylint: enable=unused-import
"""
"""Functions for downloading and reading MNIST data. END"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入x向量 None 表示张量的第一个维度可以为任意长
x = tf.placeholder(tf.float32, [None, 784])

# 784维的图像像素值，10维的输出证据值向量
w = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

# 计算交叉熵 H = -E(pk*log(1/qk)) (其中pk为真实分布，qk为非真实分布)
y_ = tf.placeholder("float", [None, 10])
# y是我们预测的概率分布, y_是实际的分布（我们输入的one-hot vector)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 采用梯度下降法，0.01的学习速率 来训练，用梯度下降算法训练你的模型，微调你的变量，不断减少成本。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 模型训练1000次(每次循环只抓取了100个数据进行训练，因此是随机梯度下降)
for i in range(1000):
    # 每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，
    # 然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# tf.arg_max用于返回y数组中其中数值最大的值所在的索引
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
print("correct_prediction: %s" % correct_prediction)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("训练后模型的准确率为：%s" % result)


