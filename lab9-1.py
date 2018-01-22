import tensorflow as tf

'''
1. XOR을 logistic regression으로 구하기.  
learning_rate = 0.1일 경우 0.5의 Accuracy가 나오는데 0.01일 경우 0.75가 나옴. hypothesis를 그려보고싶다.
step을 10000을 돌리니깐 결국 Accuracy가 0.5로 돌아감. 이는 위의 경우가 overfitting 된 가능성이 높다.
'''

'''
x_data = [
		[0,0], 
		[0,1], 
		[1,0], 
		[1,1]
	]

y_data = [
		[0],
		[1],
		[1],
		[0]
	]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(X, W)+b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(10001):
		sess.run(train, feed_dict = {X:x_data, Y:y_data})
		if step % 1000 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})

	print("hypothesis : \n", h, "\nCorrect : \n", c, "\nAccuracy : \n", a)

'''

'''
2. XOR을 Neural Net으로 구하기. (wide = 2, deep = 2)
이상태에서 step을 10000, learning_rate을 0.1로 할 경우 충분한 학습이 안되어 정확하게 결과를 예측하지 못하는 결과가 발생한다.
따라서 step을 늘리고 learning_rate을 줄이므로써 극복 가능한데 오랜시간이 걸리므로 좋은 방법이 아니다. 
그래서 wide와 deep의 값을 조정해보는 것이 좋다.

'''

"""
x_data = [
		[0,0], 
		[0,1], 
		[1,0], 
		[1,1]
	]

y_data = [
		[0],
		[1],
		[1],
		[0]
	]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 2]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([2]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(10001):
		sess.run(train, feed_dict = {X:x_data, Y:y_data})
		if step % 1000 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2))

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})

	print("hypothesis : \n", h, "\nCorrect : \n", c, "\nAccuracy : \n", a)
"""

'''
3. XOR을 Neural Net으로 구하기. (wide = 10, deep = 2)
wide를 크게하니깐 cost도 기하급수적으로 작아지며 hypothesis도 답에 근사하게 접근한다.

'''

'''
x_data = [
		[0,0], 
		[0,1], 
		[1,0], 
		[1,1]
	]

y_data = [
		[0],
		[1],
		[1],
		[0]
	]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 10]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([10, 1]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(10001):
		sess.run(train, feed_dict = {X:x_data, Y:y_data})
		if step % 1000 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})

	print("hypothesis : \n", h, "\nCorrect : \n", c, "\nAccuracy : \n", a)

'''

'''
4. XOR을 Neural Net으로 구하기. (wide = 10, deep = 4)


'''

x_data = [
		[0,0], 
		[0,1], 
		[1,0], 
		[1,1]
	]

y_data = [
		[0],
		[1],
		[1],
		[0]
	]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
wide, deep = 10, 4

# list 형태로 만듬 
W_list = [tf.Variable(tf.random_normal([2, wide]))]
b_list = [tf.Variable(tf.random_normal([wide]))]
layer_list = [tf.sigmoid(tf.matmul(X, W_list[-1])+b_list[-1])]

for deep in range(deep-2):
	W_list.append(tf.Variable(tf.random_normal([wide, wide])))
	b_list.append(tf.Variable(tf.random_normal([wide])))
	layer_list.append(tf.sigmoid(tf.matmul(layer_list[-1], W_list[-1])+b_list[-1]))

W4 = tf.Variable(tf.random_normal([wide, 1]), name = 'weight4')
b4 = tf.Variable(tf.random_normal([1]), name = 'bias4')
hypothesis = tf.sigmoid(tf.matmul(layer_list[-1], W4)+b4)


'''
# 원래 형태
W1 = tf.Variable(tf.random_normal([2, 10]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([10, 10]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([10]), name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2)+b2)

W3 = tf.Variable(tf.random_normal([10, 10]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([10]), name = 'bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3)+b3)

W4 = tf.Variable(tf.random_normal([wide, 1]), name = 'weight4')
b4 = tf.Variable(tf.random_normal([1]), name = 'bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4)+b4)
'''

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(10001):
		sess.run(train, feed_dict = {X:x_data, Y:y_data})
		if step % 1000 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})

	print("hypothesis : \n", h, "\nCorrect : \n", c, "\nAccuracy : \n", a)

