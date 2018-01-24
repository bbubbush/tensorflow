import tensorflow as tf
import matplotlib.pyplot as plt

'''
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

#Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
W_val = []
cost_val = []
for i in range(-30, 50):
	feed_W = i * 0.1
	curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
	W_val.append(curr_W)
	cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val, cost_val)
plt.show()
'''
# 수동으로 Gradient Descent Algorithm을 구현
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W = tf.Variable(tf.random_normal([1]), name = 'weight')
W = tf.Variable(5.)
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
X = [1, 2, 3]
Y = [1, 2, 3]

hypothesis = X * W

# cost = tf.reduce_sum(tf.square(hypothesis - Y))
cost = tf.reduce_mean(tf.square(hypothesis - Y))


'''
# Minimize : Gradient Descent using derivative: W := learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)	# tf에서는 equal 대입이 아니라 assign을 사용해야 함
'''

# 위 4줄의 수동으로 작성한 코드를 tf는 이렇게 제공함
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(100):
	print(step, sess.run(W))
	sess.run(train)
	'''
	sess.run(update, feed_dict = {X:x_data, Y:y_data})
	print(step, sess.run(cost, feed_dict = {X:x_data, Y:y_data}), sess.run(W))
	'''

# gradient를 임의로 수정하는 방법도 있음
gradient = tf.reduce_mean((W * X - Y) * X) * 2
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
gvs = optimizer.compute_gradients(cost)	# 이렇게 받아서 원하는 데이터 가공을 한 후 아래 코드로 넣어주면 됨
apply_gradients = optimizer.apply_gradients(gvs)

sess.run(tf.global_variables_initializer())
for step in range(100):
	print(step, sess.run([gradient, W, gvs]))
	sess.run(apply_gradients)
