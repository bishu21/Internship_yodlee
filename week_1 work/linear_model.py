# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:30:31 2018

@author: bchaudhary
"""

import tensorflow as tf


train_input=[[0, 0],[0,1],[1,0],[1,1]]

import numpy as np
from random import shuffle
#train_input = np.array(train_input,tf.float32)

NUM_EXAMPLES=10

train_input = ['{0:020b}'.format(i) for i in range(2**3)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]

ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append(j)
    ti.append(np.array(temp_list))
train_input = ti
#print(train_input[:2])

tf.to_float(train_input)

train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j== 0:
            count+=1
    temp_list = ([0]*1)
    if(count):
        temp_list[0]=0
    else :
        temp_list[0]=1
    train_output.append(temp_list)

#print(train_output[:2])
#print(train_input.shape)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]


#tf.reset_default_graph()

w= tf.Variable(tf.ones(20),tf.float32)
#w=tf.transpose(w)
b=tf.Variable([.3],tf.float32)


#
#x=tf.placeholder(tf.float32,shape=[1,20])
##x=tf.transpose(x)
#
#w =tf.reshape(w,[20,1])
#
#
#
#linear_model = tf.matmul(x,w) +b
#
#y=tf.placeholder(tf.float32)
#
#squared = tf.square(linear_model-y)
#loss = tf.reduce_sum(squared)
#
#optimizer = tf.train.GradientDescentOptimizer(0.01)
#train = optimizer.minimize(loss)
#
#init = tf.initialize_all_variables()
#
#sess = tf.Session()
#
#sess.run(init)
#
#for i in range(1000):
#    for j in range(10):
#        inp = train_input[ptr:ptr+j]
#        out = train_output[ptr:ptr+j]
        #sess.run(train,{x:[inp],y:out})

#print(sess.run([w,b]))

#print(sess.run(linear_model,{x:[[0,0]]}))
