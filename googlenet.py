'''
googlenet的网络结构
sstart at 2019.3.19
'''
 
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops
 
 
# 定义Inception层
def get_inception_layer(inputs, conv11_size, conv33_11_size, conv33_size, conv55_11_size, conv55_size, pool_size):
    with tf.variable_scope("conv_1*1"):
        conv11 = layers.conv2d(inputs, conv11_size, [1, 1])
    with tf.variable_scope("conv_3*3"):
        conv33_11 = layers.conv2d(inputs, conv33_11_size, [1, 1])
        conv33 = layers.conv2d(conv33_11, conv33_size, [3, 3])
    with tf.variable_scope("conv_5*5"):
        conv55_11 = layers.conv2d(inputs, conv55_11_size, [1, 1])
        conv55 = layers.conv2d(conv55_11, conv55_size, [5, 5])
    with tf.variable_scope("pool"):
        pool = layers.max_pool2d(inputs, [3, 3], stride=1)
        pool11 = layers.conv2d(pool, pool_size, [1, 1])
    return tf.concat([conv11, conv33, conv55, pool11], 3)
 
 
# 定义网络中的辅助分类器(softmax)
def aux_logit_layer(inputs, n_classes, is_training):
    with tf.variable_scope('ave_pool'):
        ave_pool1 = layers.avg_pool2d(inputs, [5, 5], stride=3)
    with tf.variable_scope("conv11"):
        conv11 = layers.conv2d(ave_pool1, 128, [1, 1])
    with tf.variable_scope("flatten"):
        flat = tf.reshape(conv11, [-1, 2048])
    with tf.variable_scope("fc1"):  # activation_fn=None激活函数默认为relu函数
        fc = layers.fully_connected(flat, 2048, activation_fn=None)
    with tf.variable_scope("drop"):
        drop = layers.dropout(fc, 0.3, is_training=is_training)
    with tf.variable_scope("linear"):
        linear = layers.fully_connected(drop, n_classes, activation_fn=None)
    with tf.variable_scope("soft"):
        soft = tf.nn.softmax(linear)
    return soft
 
 
def googlenet(inputs, rate=0.4, n_classes=10):
    with tf.name_scope('googlenet'):
        conv1 = tf.nn.relu(layers.conv2d(inputs, 64, [7, 7], stride=2, scope='conv1'))
        pool1 = layers.max_pool2d(conv1, [3, 3], scope='pool1')
        conv2 = tf.nn.relu(layers.conv2d(pool1, 192, [3, 3], stride=1, scope='conv2'))
        pool2 = layers.max_pool2d(conv2, [3, 3], stride='pool2')
 
        with tf.variable_scope('Inception_3a'):
            incpt3a = get_inception_layer(pool2, 64, 96, 128, 16, 32, 32)
        with tf.variable_scope("Inception_3b"):
            incpt3b = get_inception_layer(incpt3a, 128, 128, 192, 96, 64)
        pool3 = layers.max_pool2d(incpt3b, [3, 3], scope='pool3')
        with tf.variable_scope("Inception_4a"):
            incpt4a = get_inception_layer(pool3, 192, 96, 208, 16, 48, 64)
        with tf.variable_scope("aux_logit_layer1"):
            aux1 = aux_logit_layer(incpt4a, n_classes, is_training=True)
        with tf.variable_scope("Inception_4b"):
            incpt4b = get_inception_layer(incpt4a, 160, 112, 224, 24, 64, 64)
        with tf.variable_scope("Inception_4c"):
            incpt4c = get_inception_layer(incpt4b, 128, 128, 256, 24, 64, 64)
        with tf.variable_scope("Inception_4d"):
            incpt4d = get_inception_layer(incpt4c, 112, 144, 288, 32, 64, 64)
        with tf.variable_scope("aux_logit_layer2"):
            aux2 = aux_logit_layer(incpt4d, n_classes, is_training=True)
        pool4 = layers.max_pool2d(incpt4d, [3, 3], scope='pool4')
        with tf.variable_scope("Inception_5a"):
            incept5a = get_inception_layer(pool4, 256, 160, 320, 32, 128, 128)
        with tf.variable_scope("Inception_5b"):
            incept5b = get_inception_layer(incept5a, 384, 192, 384, 48, 128, 128)
        pool5 = layers.avg_pool2d(incept5b, [7, 7], stride=1, scope='pool5')
        reshape = tf.reshape(pool5, [-1, 2048])
        drop = layers.dropout(reshape, rate, is_training=True)
        linear = layers.fully_connected(drop, n_classes, activation_fn=None, scope='linear')
        # soft = tf.nn.softmax(linear)
    return linear, aux1, aux2
 
 
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss
 
 
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
 
 
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)