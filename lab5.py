# logistic regression classifier 구현
import tensorflow as tf
import numpy as np


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='b')

# hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b)) 와 아래는 같은 수식
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(10001):
		cost_val, _ = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
		if step % 2000 == 0:
			print(step, cost_val)

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})

	print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)

	# 학습된 데이터를 활용하여 데이터를 예측하는 내용
	x_val = [[2.5, 2.5], [0, 0], [1, 1], [1,4], [4,4]]
	h1, c1 = sess.run([hypothesis, predicted], feed_dict={X: x_val})
	print("\nHypothesis:\n", h1, "\n\nPredicted (Y):\n", c1)


'''
# 당뇨병 데이터를 가지고 실습
xy = np.loadtxt('data-03-diabetes.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:,0:-1]
y_data = xy[:, [-1]]
print(x_data.shape)
print(y_data.shape)


X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='b')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(10001):
		cost_val, _ = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
		if step % 200 == 0:
			print(step, cost_val)

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})

	print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)

'''

