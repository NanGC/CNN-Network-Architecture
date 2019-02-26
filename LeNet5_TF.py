<<<<<<< HEAD
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sys
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
logs_train_dir = './Model'

def weight_variable(shape):
    # 产生正态分布，标准差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(images):
    # 第1层卷积
    with tf.variable_scope('conv1'):
        W_conv1 = tf.Variable(weight_variable([5, 5, 1, 6]), name="weight")
        b_conv1 = tf.Variable(bias_variable([6]), name="bias")
        h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    # print(np.shape(h_conv1))
    # 第2层池化
    with tf.variable_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv1)
    # print(np.shape(h_pool2))

    # 第3层卷积
    with tf.variable_scope('conv3'):
        W_conv3 = tf.Variable(weight_variable([5, 5, 6, 16]), name="weight")
        b_conv3 = tf.Variable(bias_variable([16]), name="bias")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # print(np.shape(h_conv3))
    # 第4层池化
    with tf.variable_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv3)
    # print(np.shape(h_pool4))
    h_pool4_flat = tf.reshape(h_pool4, [-1, 7 * 7 * 16])

    # 5、6、7全连接层
    with tf.variable_scope('fc5'):
        W_fc5 = weight_variable([7 * 7 * 16, 120])
        b_fc5 = bias_variable([120])
        h_fc5 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc5) + b_fc5)

    with tf.variable_scope('fc6'):
        W_fc6 = weight_variable([120, 84])
        b_fc6 = bias_variable([84])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    with tf.variable_scope('out'):
        W_out = weight_variable([84, 10])
        b_out = bias_variable([10])
        h_out = tf.nn.softmax(tf.matmul(h_fc6, W_out) + b_out)

    # h_out_drop = tf.nn.dropout(h_out, 0.5)
    return h_out

x = tf.placeholder(tf.float32, [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

y_conv = inference(x_image)

# 定义损失函数和学习步骤
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))   # 交叉熵
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)   # 用Adam优化器

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 这个是log汇总记录
summary_op = tf.summary.merge_all()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # summary_str = sess.run(summary_op)
        # train_writer.add_summary(summary_str, i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
saver.save(sess, checkpoint_path)
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
=======
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sys
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
logs_train_dir = './Model'

def weight_variable(shape):
    # 产生正态分布，标准差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(images):
    # 第1层卷积
    with tf.variable_scope('conv1'):
        W_conv1 = tf.Variable(weight_variable([5, 5, 1, 6]), name="weight")
        b_conv1 = tf.Variable(bias_variable([6]), name="bias")
        h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    # print(np.shape(h_conv1))
    # 第2层池化
    with tf.variable_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv1)
    # print(np.shape(h_pool2))

    # 第3层卷积
    with tf.variable_scope('conv3'):
        W_conv3 = tf.Variable(weight_variable([5, 5, 6, 16]), name="weight")
        b_conv3 = tf.Variable(bias_variable([16]), name="bias")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # print(np.shape(h_conv3))
    # 第4层池化
    with tf.variable_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv3)
    # print(np.shape(h_pool4))
    h_pool4_flat = tf.reshape(h_pool4, [-1, 7 * 7 * 16])

    # 5\6\7全连接层
    with tf.variable_scope('fc5'):
        W_fc5 = weight_variable([7 * 7 * 16, 120])
        b_fc5 = bias_variable([120])
        h_fc5 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc5) + b_fc5)

    with tf.variable_scope('fc6'):
        W_fc6 = weight_variable([120, 84])
        b_fc6 = bias_variable([84])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    with tf.variable_scope('out'):
        W_out = weight_variable([84, 10])
        b_out = bias_variable([10])
        h_out = tf.nn.softmax(tf.matmul(h_fc6, W_out) + b_out)

    # h_out_drop = tf.nn.dropout(h_out, 0.5)
    return h_out

x = tf.placeholder(tf.float32, [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

y_conv = inference(x_image)

# 定义损失函数和学习步骤
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))   # 交叉熵
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)   # 用Adam优化器

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 这个是log汇总记录
summary_op = tf.summary.merge_all()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # summary_str = sess.run(summary_op)
        # train_writer.add_summary(summary_str, i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
saver.save(sess, checkpoint_path)
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
>>>>>>> 0c8196c6ef2fb0be7b50fc2be379a6bbc3981ab0
