#! /usr/bin/python3
# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/6/2

import tensorflow as tf;


def save():
    # Create some variables.
    v1 = tf.Variable(tf.constant(1.0), name="v1")
    v2 = tf.Variable(tf.constant(2.0), name="v2")
    # Add an op to initialize the variables.

    add_opt = tf.add(v1, v2)
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        v3 = sess.run(add_opt)
        print("v3:", v3)
        # Do some work with the model.
        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: ", save_path)


def restore():
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/tmp/model.ckpt")


if __name__ == '__main__':
    save()
    restore()
