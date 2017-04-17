import tensorflow as tf
import numpy as np

def model_NiN(data, reuse=False, train=True):
	'''
	conv1_1 = tf.layers.conv2d(inputs=data, filters=192, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
	conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=160, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
	conv1_3 = tf.layers.conv2d(inputs=conv1_2, filters=96, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1_3, pool_size=[3, 3], strides=2)

	if train:
		pool1 = tf.nn.dropout(pool1, 0.5)

	conv2_1 = tf.layers.conv2d(inputs=pool1, filters=192, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
	conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=192, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
	conv2_3 = tf.layers.conv2d(inputs=conv2_2, filters=192, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
	pool2 = tf.contrib.layers.avg_pool2d(inputs=conv2_3, kernel_size=[3, 3], stride=2)

	if train:
		pool2 = tf.nn.dropout(pool2, 0.5)

	conv3_1 = tf.layers.conv2d(inputs=pool2, filters=192, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
	conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=192, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
	conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=10, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
	pool3 = tf.contrib.layers.avg_pool2d(inputs=conv3_3, kernel_size=[8, 8], stride=1)

	logits = tf.layers.dense(inputs=pool3, units=10)

	return logits
	'''
	with tf.variable_scope('NiN', reuse=reuse):
		data = tf.reshape(data, shape=[-1, 28, 28, 1])

		conv_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
		zero_initializer = tf.truncated_normal_initializer(stddev=0.0001)
		conv1_1_w = tf.get_variable('conv1_1_w', [5, 5, 1, 192], initializer=conv_initializer)
		conv1_1_b = tf.get_variable('conv1_1_b', [192], initializer=zero_initializer)
		conv1_2_w = tf.get_variable('conv1_2_w', [1, 1, 192, 160], initializer=conv_initializer)
		conv1_2_b = tf.get_variable('conv1_2_b', [160], initializer=zero_initializer)
		conv1_3_w = tf.get_variable('conv1_3_w', [1, 1, 160, 96], initializer=conv_initializer)
		conv1_3_b = tf.get_variable('conv1_3_b', [96], initializer=zero_initializer)

		conv2_1_w = tf.get_variable('conv2_1_w', [5, 5, 96, 192], initializer=conv_initializer)
		conv2_1_b = tf.get_variable('conv2_1_b', [192], initializer=zero_initializer)
		conv2_2_w = tf.get_variable('conv2_2_w', [1, 1, 192, 192], initializer=conv_initializer)
		conv2_2_b = tf.get_variable('conv2_2_b', [192], initializer=zero_initializer)
		conv2_3_w = tf.get_variable('conv2_3_w', [1, 1, 192, 192], initializer=conv_initializer)
		conv2_3_b = tf.get_variable('conv2_3_b', [192], initializer=zero_initializer)

		conv3_1_w = tf.get_variable('conv3_1_w', [3, 3, 192, 192], initializer=conv_initializer)
		conv3_1_b = tf.get_variable('conv3_1_b', [192], initializer=zero_initializer)
		conv3_2_w = tf.get_variable('conv3_2_w', [1, 1, 192, 160], initializer=conv_initializer)
		conv3_2_b = tf.get_variable('conv3_2_b', [160], initializer=zero_initializer)
		conv3_3_w = tf.get_variable('conv3_3_w', [1, 1, 160, 10], initializer=conv_initializer)
		conv3_3_b = tf.get_variable('conv3_3_b', [10], initializer=zero_initializer)
		
		conv1_1 = tf.nn.conv2d(data, conv1_1_w, strides=[1, 1, 1, 1], padding="SAME")
		relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_b))
		conv1_2 = tf.nn.conv2d(relu1_1, conv1_2_w, strides=[1, 1, 1, 1], padding="SAME")
		relu1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, conv1_2_b))
		conv1_3 = tf.nn.conv2d(relu1_2, conv1_3_w, strides=[1, 1, 1, 1], padding="SAME")
		relu1_3 = tf.nn.relu(tf.nn.bias_add(conv1_3, conv1_3_b))
		pool1 = tf.nn.max_pool(relu1_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
		if train:
			pool1 = tf.nn.dropout(pool1, 0.5)

		conv2_1 = tf.nn.conv2d(pool1, conv2_1_w, strides=[1, 1, 1, 1], padding="SAME")
		relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_b))
		conv2_2 = tf.nn.conv2d(relu2_1, conv2_2_w, strides=[1, 1, 1, 1], padding="SAME")
		relu2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, conv2_2_b))
		conv2_3 = tf.nn.conv2d(relu2_2, conv2_3_w, strides=[1, 1, 1, 1], padding="SAME")
		relu2_3 = tf.nn.relu(tf.nn.bias_add(conv2_3, conv2_3_b))
		pool2 = tf.nn.avg_pool(relu2_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
		if train:
			pool2 = tf.nn.dropout(pool2, 0.5)

		conv3_1 = tf.nn.conv2d(pool2, conv3_1_w, strides=[1, 1, 1, 1], padding="SAME")
		relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, conv3_1_b))
		conv3_2 = tf.nn.conv2d(relu3_1, conv3_2_w, strides=[1, 1, 1, 1], padding="SAME")
		relu3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2, conv3_2_b))
		conv3_3 = tf.nn.conv2d(relu3_2, conv3_3_w, strides=[1, 1, 1, 1], padding="SAME")
		relu3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3, conv3_3_b))
		pool3 = tf.nn.avg_pool(relu3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

		flatten = tf.contrib.layers.flatten(pool3)
		#sm = tf.nn.softmax(pool3)
		print(flatten)
		w = [conv1_1_w, conv1_1_b,
			 conv1_2_w, conv1_2_b,
			 conv1_3_w, conv1_3_b,
			 conv2_1_w, conv2_1_b,
			 conv2_2_w, conv2_2_b,
			 conv2_3_w, conv2_3_b,
			 conv3_1_w, conv3_1_b,
			 conv3_2_w, conv3_2_b,
			 conv3_3_w, conv3_3_b]

		#return sm, w
		return flatten, w


		
