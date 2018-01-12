import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7	# Y의 label이 7가지가 나오기 때문

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])	# float32로 주면 에러가 남

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])	# r+1 된 차원을 다시 r차원으로 낮추는 과정. 자주 사용됨

W = tf.Variable(tf.random_normal([16, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

# 두 코스트는 같은 값을 갖는다
#cost = tf.reduce_mean(-tf.reduce_sum(Y_one_hot * tf.log(hypothesis), axis = 1)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)

correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(2001):
		sess.run(optimizer, feed_dict = {X : x_data, Y : y_data})
		if step % 100 == 0 :
			loss, acc = sess.run([cost, accuracy], feed_dict = {X : x_data, Y : y_data})
			print("step : {:5}\tLoss : {:.3f}\tAcc : {:.2%}".format(step, loss, acc))	# 정렬을 위해 왼쪽에 공백을 주는 방식으로 출력

	pred = sess.run(prediction, feed_dict = {X : x_data})

	for p, y in zip(pred, y_data.flatten()):
		print("[{}] Prediction : {} True Y : {}".format(p == int(y), p, int(y)))
