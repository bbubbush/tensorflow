import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)	# for reproducibility (재현성)

# download data_sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10 	# label = 0~9

# image pixel is (28 * 28) = 784
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# use cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))		# if axis is 0 then return values is sum columns. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))	# tf.equal's return type is bool. True is , False is 0
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.int32) * 100)	# bool type change to float32. but change to int32 is fine if multiply 100.
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 와 위의 코드는 같다.

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples / batch_size)

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			c, _ = sess.run([cost, optimizer], feed_dict = {X: batch_xs, Y: batch_ys})
			avg_cost += c / total_batch

		print('Epoch:', '%02d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

	print("Learning Finished!")

	'''
	accuracy.eval은 sess.run을 활용할 때, 넘기는 값이 1개일 때 간단하게 사용할 수 있다.
	sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})와 동일하게 사용 가능
	'''
	print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

	# 0부터 num_examples의 전체 개수 사이에서 랜덤하게 하나 선택. size인자를 넣으면 배열형태로 return 가능
	r = random.randint(0, mnist.test.num_examples - 1)	
	print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))	# True data

	# mnist.test.images를 직접 보고 싶은데 어디서 찾아야하는지 모르겠다. 
	print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))	# Prediction data

	plt.imshow(
		mnist.test.images[r:r+1].reshape(28, 28),
		cmap='Blues',
		interpolation='nearest'
		)
	plt.show()
