import tensorflow as tf
# 기존의 방법으로 linear regression을 만드는 방법
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y:y_data})

	if step % 20000 == 0:
		print(step, 'Cost:', cost_val, '\nPridiction:\n', hy_val)

# x변수의 개수가 늘어날수록 코드가 지저분하고 스파게티 코드가 될 가능성이 높은 단점이 있어 Matrix를 사용하여 처리한다.
# loop의 반복이 많이면 많을수록 도달하려는 값과의 오차가 점점 적어진다.
x_data = [
  [73., 80., 75.]
, [93., 88., 93.]
, [89., 91., 90.]
, [96., 98., 100.]
, [73., 66., 70.]
]

y_data = [
  [152.]
, [185.]
, [180.]
, [196.]
, [142.]
]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(20001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

	if step % 20000 == 0:
		print(step, 'Cost:', cost_val, '\nPridiction:\n', hy_val)

