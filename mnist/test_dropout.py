# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/5/6

import unittest
import tensorflow as tf


class dropoutTest(unittest.TestCase):

    def test_dropout(self):
        # 原始变量：
        # [[1. 1. 1. 1.]
        # [1. 1. 1. 1.]]
        # [1. 1. 1. 1.]]
        # [1. 1. 1. 1.]]
        # 输出变量：
        # [[0.  0.  0.  2.5]
        #  [0.  2.5 2.5 0.]
        # [2.5 2.5  0. 0.]
        # [2.5 0.  2.5 2.5]]

        # 结论分析(具体参考dropoutTest测试用例):
        # 1、说下所有的输入元素N个中，有N*keep_prob个元素会修改值为当前值得1/keep_prob倍
        # 2、其他元素则设置为0
        # 3、具体哪些元素扩大1/keep_prob倍，哪些元素为0，随机决定
        keep_prob = tf.placeholder(tf.float32)
        x = tf.Variable(tf.ones([4, 4]))
        y = tf.nn.dropout(x, keep_prob)

        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)

        print("\nx == \n%s\n" % x.eval())
        print(sess.run(y, feed_dict={keep_prob: 0.4}))

