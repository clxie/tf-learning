# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/6/3
# 更加详细的图表展示，请参考：mnist中的mnist_experts.py使用示例

import tensorflow as tf


def create_tf_board():
    a = tf.constant([1, 2], name="a")
    b = tf.constant([3, 4], name="b")
    c = tf.add(a, b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("/tmp/tensorboard/add", sess.graph)


if __name__ == '__main__':
    create_tf_board()
