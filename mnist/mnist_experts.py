# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/4/12
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])


# ----------------------------- 构造多层神经网络来改善训练效果-------------------------------
def weight_variable(shape):
    # 截断正态分布，标准差为0.1(从截断的正态分布中输出随机值)
    # 生成的值服从具有指定平均值和标准偏差的正态分布，
    # 如果生成的值大于平均值2个标准偏差的值则丢弃重新选择
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    # 构造值为0.1的多维shape矩阵
    constant = tf.constant(0.1, shape=shape)
    return tf.Variable(constant)


def conv2d(x, W):
    # x：input,指需要做卷积的输入图像，它要求是一个Tensor，
    #   具有[batch, in_height, in_width, in_channels]这样的shape，
    #   具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，
    #   要求类型为float32和float64其中之一
    # W: filter, 相当于CNN中的卷积核，它要求是一个Tensor，
    #   具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
    #   具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
    #   有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    # padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
    #   SAME: 表示经过卷积过滤之后的输出图像与输入保持维度一致，不够的，则在原始图像边缘填充0
    # 返回值：返回一个Tensor，这个输出，就是我们常说(卷积后)的feature map
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# max pooling 的操作如下图所示：整个图片被不重叠的分割成若干个同样大小的小块（pooling size）。
# 每个小块内只取最大的数字，再舍弃其他节点后，保持原有的平面结构得出 output。
# Max pooling 的主要功能是 downsampling，却不会损坏识别结果。 这意味着卷积后的 Feature Map 中有对于识别物体不必要的冗余信息。
# ** 更多max_pool请参考博客：http://www.techweb.com.cn/network/system/2017-07-13/2556494.shtml
def max_pool_2x2(x):
    # x: value,需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
    #   依然是[batch, height, width, channels]这样的shape
    # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    #   因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    # padding：和卷积类似，可以取’VALID’ 或者’SAME’
    # 返回值：一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

sess.run(tf.global_variables_initializer())
print(b_conv1.eval())

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)) + b_conv1
h_max_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_max_pool1, W_conv2)) + b_conv2
h_max_pool2 = max_pool_2x2(h_conv2)

# 全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_max_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练，评估模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("多层神经网络改善训练模型的准确率为 %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
