from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST/',one_hot=True)

##1.定义占位符
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,10])
x_img = tf.reshape(x,[-1,28,28,1])

##2.初始化参数
def weiht_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') #strides限定步长，一般1,4位都是1
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
##第一层卷积
W_conv1 = weiht_variable([5,5,1,32])   ##5*5的filter
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_img,W_conv1)+b_conv1) ##28*28*32  (same填充)
h_pool1 = max_pool_2x2(h_conv1)                     ##14*14*32

##第二层卷积
W_conv2 = weiht_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) ##14*14*64
h_pool2 = max_pool_2x2(h_conv2)                       ##7*7*64

##全连接
w_fc1 = weiht_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

##droupout
keep_drop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_drop)

##输出层
w_fc2 = weiht_variable([1024,10])
b_fc2 = bias_variable([10])
yHat = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

##计算损失
loss = -tf.reduce_sum(y*tf.log(yHat))
optimizer = tf.train.AdadeltaOptimizer(1e-2)
train = optimizer.minimize(loss)

##评价指标
Right_predict = tf.equal(tf.argmax(yHat,1),tf.argmax(y,1))
Accuracy = tf.reduce_mean(tf.cast(Right_predict,tf.float32))

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    for step in range(5000):
        batch_x,batch_y = mnist.train.next_batch(100)
        train.run(feed_dict={x:batch_x,y:batch_y,keep_drop:0.8})
        if step%100 == 0:
            print('第%d次迭代的准确率为%f'%(step,Accuracy.eval(feed_dict={x:batch_x,y:batch_y,keep_drop:1})))
    print('测试集中，准确率为%f'%(Accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_drop:1})))
