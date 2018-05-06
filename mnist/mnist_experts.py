# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/4/12
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


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


# Max pooling 的主要功能是 downsampling，却不会损坏识别结果。 这意味着卷积后的 Feature Map 中有对于识别物体不必要的冗余信息。
# max pooling 的操作：整个图片被不重叠的分割成若干个同样大小的小块（pooling size）。
# 每个小块内只取最大的数字，再舍弃其他节点后，保持原有的平面结构得出 output。
# ** 更多max_pool请参考博客：http://www.techweb.com.cn/network/system/2017-07-13/2556494.shtml
def max_pool_2x2(x):
    # x: value,需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
    #   依然是[batch, height, width, channels]这样的shape
    # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    #   因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    # padding：和卷积类似，可以取’VALID’ 或者’SAME’
    # 返回值：一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    # 返回值的矩阵大小相对于原始矩阵，降维到strides分之倍
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 输入x向量 None 表示张量的第一个维度可以为任意长
x = tf.placeholder(tf.float32, [None, 784])
# 计算交叉熵 H = -E(pk*log(1/qk)) (其中pk为真实分布，qk为非真实分布)
y_ = tf.placeholder("float", [None, 10])

# ** 第一层卷积 **
# 初始化为shape的随机正态分布矩阵变量
# 其中shape=[5, 5, 1, 32]，从最右边数字开始（为32列），最里面1行32列，接着外面包装5维，再5维
# 由于W_conv1卷积h核的最后一维是32，因此输出也将卷积值重复32维输出
W_conv1 = weight_variable([5, 5, 1, 32])
# 初始化32列变量（值为0.1）
b_conv1 = bias_variable([32])

sess.run(tf.global_variables_initializer())
print(W_conv1.eval())
print(b_conv1.eval())

# 将x变量，reshape为28维包装的28行1列(28*28=784)
# 其中-1位自动匹配
x_image = tf.reshape(x, [-1, 28, 28, 1])
print("x_image == %s" % x_image)  # shape=(?, 28, 28, 1)

# conv2d：卷积计算，获取(卷积后)的feature map（因为卷积设置的SAME，因此输出与x_image一致）
# 输入图像x_image：[batch, in_height, in_width, in_channels]
# 卷积核filter，W_conv1：[filter_height, filter_width, in_channels, out_channels]
# relu：激活函数
conv2d_value = conv2d(x_image, W_conv1)
print("\nconv2d_value == %s" % (conv2d_value))  # shape=(?, 28, 28, 32)
h_conv1 = tf.nn.relu(conv2d_value) + b_conv1
print("\nh_conv1 == %s" % (h_conv1))  # shape=(?, 28, 28, 32)

# 池化降维，去掉图片中冗余信息（max_pool即按照ksize的矩阵模板，进行池化，选择其中值最大的项）
h_max_pool1 = max_pool_2x2(h_conv1)  # 如果步长为2*2 那么则返回的shape结果矩阵，维度下降一半
print("\nh_max_pool1 == %s" % (h_max_pool1))  # shape=(?, 14, 14, 32)

# ** 第二层卷积 **
W_conv2 = weight_variable([5, 5, 32, 64])
# 初始化64列变量（值为0.1）
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_max_pool1, W_conv2)) + b_conv2
h_max_pool2 = max_pool_2x2(h_conv2)
print("\nh_conv2 == %s" % (h_conv2))  # shape=(?, 14, 14, 64)
print("\nh_max_pool2 == %s" % (h_max_pool2))  # shape=(?, 7, 7, 64)

# 全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_max_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，我们在输出层之前加入dropout
# 更多参考：https://www.jianshu.com/p/c9f66bc8f96c
# https://blog.csdn.net/stdcoutzyx/article/details/49022443
keep_prob = tf.placeholder("float")  # 占位符，训练过程中启用dropout设置0.5，在测试过程中关闭dropout设置为1.0

# dropout结论分析(具体参考dropoutTest测试用例):
# 1、说下所有的输入元素N个中，有N*keep_prob个元素会修改值为当前值得1/keep_prob倍
# 2、其他元素则设置为0
# 3、具体哪些元素扩大1/keep_prob倍，哪些元素为0，随机决定
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

sess.run(tf.global_variables_initializer())

# 此处原始为20000步，速度太慢，修改为1000
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("多层神经网络改善训练模型的准确率为 %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
