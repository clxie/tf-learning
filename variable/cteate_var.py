#! /usr/bin/python3
# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/6/2

import tensorflow as tf


def create():
    random_var = tf.Variable(tf.random_normal([32 * 32, 200], stddev=0.1), name="random_var")
    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    print(random_var)


def init_by_other_val():
    # Create a variable with a random value.
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                          name="weights")
    # Create another variable with the same value as 'weights'.
    w2 = tf.Variable(weights.initialized_value(), name="w2")
    # Create another variable with twice the value of 'weights'
    w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")
    print(w_twice)


if __name__ == '__main__':
    create()
    init_by_other_val()
