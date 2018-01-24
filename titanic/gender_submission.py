import tensorflow as tf
import numpy as np
x_data = np.genfromtxt('gender_submission.csv',usecols=(0),delimiter=',',dtype=np.str, skip_header=1)
y_data = np.genfromtxt('gender_submission.csv',usecols=(1),delimiter=',',dtype=np.float32, skip_header=1)

for x, y in zip(x_data, y_data):
	print("{}\t{}".format(x, y == 1 and "생존" or "사망"))




