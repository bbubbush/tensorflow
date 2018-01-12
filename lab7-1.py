import tensorflow as tf
import numpy as np
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 3])	# Y = tf.placeholder(tf.int32, shape = [None, 3])
W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))



logits = tf.matmul(X, W) + b


hypothesis = tf.nn.softmax(logits)

'''
Y_one_hot = tf.one_hot(Y, 3)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 3])
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)
'''

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.5).minimize(cost)		# learning_rate가 클 때
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-10).minimize(cost)	# learning_rate가 작을 때
prediction = tf.argmax(hypothesis, 1)
is_corrent = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_corrent, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(2001):
		cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict = {X : x_data, Y : y_data})

		if step % 200 == 0:
			print(step, cost_val, W_val)

	print("prediction : {}".format(sess.run(prediction, feed_dict = {X : x_test})))

	print("Accuracy : {}".format(sess.run(accuracy, feed_dict = {X : x_test, Y : y_test})))
