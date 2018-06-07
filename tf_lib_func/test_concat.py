# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/6/7

import tensorflow as tf


def test_concat():
    t1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    t2 = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    print("\n t1 == %s" % (t1))
    concat0 = tf.concat([t1, t2], 0)
    concat1 = tf.concat([t1, t2], 1)
    # concat2 = tf.concat([t1, t2], 2)
    with tf.Session() as sess:
        print("\n%s" % sess.run(concat0))
        print("\n%s" % sess.run(concat1))
        # print("\n%s" % sess.run(concat2))


def test_concat1():
    # 三维数据（外层几个括号，则为几维数据）
    t1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    t2 = [[[11, 12], [13, 14]], [[15, 16], [17, 18]]]
    print("\n t1 == %s" % (t1))
    print("\n t2 == %s" % (t2))
    concat0 = tf.concat([t1, t2], 0)
    concat1 = tf.concat([t1, t2], 1)
    concat2 = tf.concat([t1, t2], 2)
    with tf.Session() as sess:
        run0 = sess.run(concat0)
        print("concat0：\n%s" % run0)
        print("concat1：\n%s" % sess.run(concat1))
        print("concat2：\n%s" % sess.run(concat2))


if __name__ == '__main__':
    # test_concat()
    test_concat1()
