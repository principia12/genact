import tensorflow as tf
import numpy as np

def model_NiN(data, train=True):
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
