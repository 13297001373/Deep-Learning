'''
这段很短的 Python 程序生成了一些三维数据, 然后用一个平面拟合它.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
#1.产生数据
x = np.float32(np.random.rand(2,100))
y = np.dot([0.1,0.2],x) + 0.3

#2.构造线性函数
W = tf.Variable(tf.random_uniform([1,2],-1,1))  ##随机初始化
b = tf.Variable(tf.zeros([1]))
yhat = tf.matmul(W,x)+b
#3.计算损失
loss = tf.reduce_mean(tf.square(y-yhat))  ##MSE
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#4.初始化变量
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1,201):
        sess.run(train)
        if step%20 == 0:
            print('第%d迭代的损失为%f'%(step,sess.run(loss)))
            print('W为',sess.run(W))

