import tensorflow as tf
import numpy as np
import pprint

tf.set_random_seed(777)

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# Simple Array
print("[ Simple Array ]")
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim)	# rank
print(t.shape)
print()

# 2D Array
print("[ 2D Array ]")
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim)
print(t.shape)
print()

# Shape, Rank, Axis
print("[ Shape, Rank, Axis ]")
t = tf.constant([1,2,3,4])
print(tf.shape(t).eval())

t = tf.constant([[1,2],
                 [3,4]])
print(tf.shape(t).eval())

t = tf.constant(
	[
		[
			[
				[1, 2, 3, 4], 
				[5, 6, 7, 8], 
				[9, 10, 11, 12]
			],
			[
				[13, 14, 15, 16], 
				[17, 18, 19, 20], 
				[21, 22, 23, 24]
			]
		]
	]
)
print(tf.shape(t).eval())
print()

# Matmul VS muliply
print("[ Matmul VS muliply ]")
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
print(tf.matmul(matrix1, matrix2).eval())
print((matrix1 * matrix2).eval())
print()

# Watch out broadcasting (Very Important)
print("[ Watch out broadcasting (Very Important) ]")
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
print((matrix1+matrix2).eval())

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
print((matrix1+matrix2).eval())
print()

# Random values for variable initializations
print("[ Random values for variable initializations ]")
print(tf.random_normal([3]).eval())
print(tf.random_uniform([2]).eval())
print(tf.random_uniform([2, 3]).eval())
print()

# Reduce mean/Sum
print("[ Reduce mean/Sum ]")
print(tf.reduce_mean([1, 2], axis=0).eval())
x = [
		[1., 0.], 
		[3., 4.]
	]
print(tf.reduce_mean(x).eval())
print(tf.reduce_mean(x, axis=0).eval())
print(tf.reduce_mean(x, axis=1).eval())
print(tf.reduce_mean(x, axis=-1).eval())
print(tf.reduce_sum(x).eval())
print(tf.reduce_sum(x, axis=0).eval())
print(tf.reduce_sum(x, axis=-1).eval())
print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval())
print()

# Argmax with axis
print("[ Argmax with axis ]")
x = [
		[0, 1, 2], 
		[2, 1, 0]
	]
print(tf.argmax(x, axis=0).eval())
print(tf.argmax(x, axis=1).eval())
print(tf.argmax(x, axis=-1).eval())

# Reshape, squeeze, expand_dims
print("[ Reshape, squeeze, expand_dims ]")
t= np.array([
				[
					[0, 1, 2], 
					[3, 4, 5]
				],
				[
					[6, 7, 8],
					[9, 10, 11]
				]
			])
print(t.shape)

print(tf.reshape(t, shape=[-1, 3]).eval())
print(tf.reshape(t, shape=[-1, 1, 3]).eval())
print(tf.squeeze([[0], [1], [2]]).eval())
print(tf.expand_dims([0, 1, 2], 1).eval())
print()

# One hot
print("[ One hot ]")
print(tf.one_hot([[0], [1], [2], [0]], depth=3).eval())
print(tf.reshape(t, shape=[-1, 3]).eval())
print()

# casting
print("[ Casting ]")
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())
print(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())
print()

# Stack
print("[ Stack ]")
x = [1, 4]
y = [2, 5]
z = [3, 6]

print(tf.stack([x, y, z]).eval())
print(tf.stack([x, y, z], axis=1).eval())
print()

# Ones like and Zeros like
print("[ Ones like and Zeros like ]")
x = [
		[0, 1, 2],
		[2, 1, 0]
	]
print(tf.ones_like(x).eval())
print(tf.zeros_like(x).eval())
print()

# Zip
print("[ Zip ]")
for x, y in zip([1, 2, 3], [4, 5, 6]):
	print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
	print(x, y, z)
print()

#Transpose
print("[ Transpose ]")
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)

t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))

t = tf.transpose(t1, [1, 0, 2])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))

t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))

t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))


