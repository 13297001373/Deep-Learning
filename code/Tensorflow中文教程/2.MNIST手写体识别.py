import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist  = input_data.read_data_sets('MNIST/',one_hot=True)
##1.定义占位符
x = tf.placeholder('float',[None,28*28])
y = tf.placeholder('float',[None,10])

##2.定义参数,softmax
W = tf.Variable(tf.random_normal([28*28,10]))
b = tf.Variable(tf.zeros([10]))

yHat = tf.nn.softmax(tf.matmul(x,W)+b)
##3.计算损失,采用交叉熵
loss = -tf.reduce_sum(y*tf.log(yHat))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

##4.初始化变量
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        #sess.run(train,feed_dict={x:batch_x,y:batch_y})
        train.run(feed_dict={x:batch_x,y:batch_y})
        if step%100 == 0:
            correct_prediction = tf.equal(tf.argmax(yHat,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            #print(accuracy.eval(feed_dict={x:batch_x,y:batch_y}))
            print('the Accuracy  is %f'%(sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels})))



