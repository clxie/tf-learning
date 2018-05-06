# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/5/6

import unittest
from mnist import mnist_experts as mnist_experts


class mnistTest(unittest.TestCase):
    @staticmethod
    def test_conv2():
        # TODO: 运行不起
        print("1")
        mnist_experts.bias_variable([1]);
        print("2")
