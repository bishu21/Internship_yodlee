import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf 

a=tf.Variable(np.ones(2))

a=tf.reshape(a,[1,2])

b = tf.constant(np.ones(2),shape=[2,1])


#b = tf.transpose(b)
c= tf.matmul(a,b)

d=tf.ones(5)

sess = tf.Session()




print(b.shape)
print(sess.run(d))
print(c.shape)