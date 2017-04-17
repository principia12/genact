from model_nin import *

from tensorflow.contrib import learn

import threading

BATCH_SIZE = 5
WEIGHT_DECAY_RATE=10
BASE_LR = 0.00001
DECAY_STEP = 10
DECAY_RATE = 0.1
MAX_EPOCH = 50

if __name__ == "__main__":
	mnist = learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images
	train_data = train_data[:500]
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images
	eval_data = eval_data[:100]
	print(train_labels[0])
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	with tf.Session() as sess:
		input_tensor_single = tf.placeholder(tf.float32, shape=(784))
		class_gt_single = tf.placeholder(tf.int32, shape=())
		q = tf.FIFOQueue(1000, [tf.float32, tf.int32], [[784], []])
		enqueue_op = q.enqueue([input_tensor_single, class_gt_single])
		input_tensor, class_gt = q.dequeue_many(BATCH_SIZE)

		input_tensor_test = tf.placeholder(tf.float32, [BATCH_SIZE, 784])

		shared_model = tf.make_template('shared_model', model_NiN)
		out_tensor, weights = shared_model(input_tensor)
		out_tensor_test, _ = shared_model(input_tensor_test, train=False)

		w_decay_loss = None
		for w in weights:
			if w_decay_loss is None:
				w_decay_loss = tf.nn.l2_loss(w)
			w_decay_loss += tf.nn.l2_loss(w)

		cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_tensor, labels=class_gt)
		loss = cls_loss + WEIGHT_DECAY_RATE*w_decay_loss
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay( BASE_LR, global_step, DECAY_STEP*len(train_data)/BATCH_SIZE, DECAY_RATE, staircase=True)
		opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

		saver = tf.train.Saver(weights, sess)
		#tf.initialize_all_variables().run()
		tf.global_variables_initializer().run()

		def load_and_enqueue(coord, train_data, train_labels, enqueue_op, input_tensor_single, class_gt_single, idx=0):
			count = 0
			length = len(train_data)
			try:
				while count < length:
					i = count % length
					input_img = train_data[i]
					out_label = train_labels[i]
					sess.run(enqueue_op, feed_dict={input_tensor_single: input_img, class_gt_single: out_label})
					count += 1
			except Exception as e:
				print("stopping...", idx, e)

		threads = []
		max_accuracy = 0.0
		max_epoch = 0
		accuracy_changed = True

		for epoch in range(0, MAX_EPOCH):
			num_thread = 20
			del threads[:]
			coord = tf.train.Coordinator()
			for i in range(num_thread):
				length = int(len(train_data)/num_thread)
				t = threading.Thread(target=load_and_enqueue, args=(coord, train_data[i*length:(i+1)*length], train_labels[i*length:(i+1)*length], enqueue_op, input_tensor_single, class_gt_single, i))
				threads.append(t)
				t.start()

			for i in range(0, len(train_data), BATCH_SIZE):
				_, l, cls_l = sess.run([opt, loss, cls_loss])
				log_file = open('log.txt', 'a')
				log_file.write("[epoch %2.4f] loss %s, cls_loss %s\n" % (epoch + float(i) / len(train_data), l, cls_l))
				log_file.close()
				print("[epoch %2.4f] loss %s, cls_loss %s\n" % (epoch + float(i) / len(train_data), l, cls_l))

			correct = 0
			for i in range(0, len(eval_data), BATCH_SIZE):
				input_images = eval_data[i: i+BATCH_SIZE]
				out_labels = eval_labels[i: i+BATCH_SIZE]

				cls_out = sess.run(out_tensor_test, feed_dict={input_tensor_test:input_images})
				out_argmax = np.argmax(cls_out, axis=-1)
				out_correct = np.sum(out_argmax == out_labels)
				log_file = open('log_test.txt', 'a')
				log_file.write("[test] correct label %d out of %d\n\n" % (out_correct, BATCH_SIZE))
				correct += out_correct
				log_file.close()
				print("[test] correct label %d out of %d" % (out_correct, BATCH_SIZE))

			accuracy = (float(correct) / len(eval_data)) * 100
			if accuracy >= max_accuracy:
				max_accuracy = accuracy
				max_epoch = epoch
				accuracy_changed = True
			else:
				accuracy_changed = False

			print("Predict Accuracy = (# of Correct Label) / (# of Test Set) = %d / %d = %.2f%%" % (
			correct, len(eval_data), accuracy))
			print("Current Maximum Accuracy = %.2f%%, epoch %d" % (max_accuracy, max_epoch))

			log_file = open('log_accuracy.txt', 'a')
			log_file.write("Predict Accuracy = (# of Correct Label) / (# of Test Set) = %d / %d = %.2f%%\n" % (
			correct, len(eval_data), accuracy))
			log_file.write("Current Maximum Accuracy = %.2f%%, epoch %d\n" % (max_accuracy, max_epoch))
			log_file.close()

			if accuracy_changed:
				saver.save(sess, 'pre_trained_%02d.ckpt'%epoch)
