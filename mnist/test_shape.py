# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/5/6

import unittest
import tensorflow as tf
import numpy as np


class shape_test_case(unittest.TestCase):

    def test_print_shape(self):
        print_shape([1, 2, 3, 4])
        print_shape(([[1, 2], [3, 4]]))

    def test_tf_constants(self):
        sess = tf.InteractiveSession()
        # 从最右边数字开始，最里面2行3列，
        # 接着外面包装4维，再二维
        init = tf.truncated_normal(shape=[2, 4, 2, 3], stddev=0.1)
        sess.run(tf.global_variables_initializer())
        print("\ninit == %s\n" % init)
        val = tf.Variable(init)
        sess.run(tf.global_variables_initializer())
        print(val.eval())

    def test_reshape(self):
        a = np.arange(32).reshape(4, 8)
        print("\na = \n%s" % a)
        a = a.reshape(8, 4)
        print("\na = \n%s" % a)
        # 4行2列，-1位自动匹配剩余维数（4）
        a = a.reshape(-1, 4, 2)
        print("\na = \n%s" % a)

    @staticmethod
    def test_conv2():
        # 图片像素为4*4
        shape = [1, 4, 4, 1];
        init = tf.truncated_normal(shape, stddev=0.1)
        x = tf.Variable(init)
        # 过滤卷积模板为2*2
        constant = tf.constant(0.1, shape=[2, 2, 1, 1])
        W = tf.Variable(constant)
        # strides步长各维都为
        out = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())
        print("\nout = \n%s" % out.eval())

    @staticmethod
    def test2_conv2():
        x = np.arange(9.0, dtype="float32").reshape([1, 3, 3, 1])
        print("\nx == \n%s" % x)
        # 过滤卷积模板为2*2 ,此处最后一位，即将之前的输出重复输出
        constant = tf.constant(1.0, shape=[2, 2, 1, 2])
        W = tf.Variable(constant)
        # strides步长各维都为
        out = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())
        print("\nout = \n%s" % out)
        print("\nout = \n%s" % out.eval())
        # out: value,需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
        #   依然是[batch, height, width, channels]这样的shape
        # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
        #   因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        # strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
        pool = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        print("\npool = \n%s" % pool)
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())
        print("\npool = \n%s" % pool.eval())


def print_shape(shape):
    print("shape == %s" % shape)
