'''
卷积神经网络的tensorflow实现
INPUT->CONV->POOL->CONV->POOL->FC->SOFTMAX
'''
import tensorflow as tf
import numpy as np
import h5py

class simple_CNN(object):

    def load_data(self,x_path,y_path):
        train_dataset = h5py.File(x_path, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

        test_dataset = h5py.File(y_path, "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

        classes = np.array(test_dataset["list_classes"][:])  # the list of classes

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        #标准化
        X_train = train_set_x_orig / 255.
        X_test = test_set_x_orig / 255.

        y_train = np.eye(6)[train_set_y_orig.reshape(-1)]
        y_test = np.eye(6)[train_set_y_orig.reshape(-1)]
        return X_train,y_train,X_test,y_test

    def create_placehold(self,n_H0,n_W0,n_C0,n_y):
        '''
        create placehold with tensorflow
        :param n_H0: scalar,height of image
        :param n_W0: scalar,width of image
        :param n_C0: scalar dim of image
        :param n_y: scalar size of output
        :return:
            X : placehold of the data input,shape[None,n_H0,n_W0,n_C0,n_y] type:'float'
            y : placehold of the data output ,shape[None,n_y] type:'float
        '''

        X = tf.placeholder(tf.float32,shape=[None,n_H0,n_W0,n_C0],name='X-input')
        y = tf.placeholder(tf.float32,shape=[None,n_y],name='y-input')
        return X,y

    def initialize_parameters(self):
        '''
        Initialize weight parameters to build a neutral network with tensorflow
        :return:
            :parameters a dictinoary of tensors contain W1,b1,W2,b2
        '''

        tf.set_random_seed(2018)


        W1 = tf.get_variable('W1',[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=2018))

        b1 = tf.get_variable('b1',[8],initializer=tf.zeros_initializer())

        W2 = tf.get_variable('W2',[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=2018))


        b2 = tf.get_variable('b2',[16],initializer=tf.zeros_initializer())

        parameters = {'W1':W1,'b1':b1,
                      'W2':W2,'b2':b2}
        return parameters
    def forword_propagation(self,X,parameters):
        '''
        Implements the forword propagation for model
        :param X: input
        :param parameters: a dictinoary of tensors contain W1,b1,W2,b2
        :return:
            Z3 output
        '''
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')+b1 ##'SAME'不是全填充，而是对于缺失的padding
        A1 = tf.nn.relu(Z1)  ##shape(61,61,8)
        P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')   ##shape(8,8,8)

        Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')+b2
        A2 = tf.nn.relu(Z2)   ##shape(7,7,16)
        P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME') ##shape(2,2,16)

        ##FLATTEN
        P2 = tf.contrib.layers.flatten(P2)  ##shape(None,2*2*16)
        Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn=None)

        return Z3

    def compute_cost(self,yHat,y):
        '''
        compute th cost
        :param yHat: predict
        :param y: real
        :return: cost
        '''

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yHat,labels=y))

        return cost
    def random_mini_batches(self,X,y,batch_size):
        m = y.shape[0]
        permutation = np.random.permutation(m)
        shuffle_X = X[permutation,:]
        shuffle_y = y[permutation,:]
        ##将数据分按batch_size的大小划分数据为多个batch
        batch_n = m // batch_size
        batchs = []
        for batch in range(1, batch_n + 2):
            start = batch_size * (batch - 1)
            end = min(batch_size*batch,m)
            batch_x = shuffle_X[start:end,:]
            batch_y = shuffle_y[start:end,:]
            batchs.append((batch_x, batch_y))

        return batchs
    def fit(self,X_train,y_train,num_epochs = 100,learning_rate=0.1,batch_size=64):
        '''

        :param X: data set, shape(None,64,64,3)
        :param y: label set,shape(None,n_y=6)
        :param num_epochs: Iterater times
        :param learning_rate: learning_rate of optimizer
        :param batch_size: mini_batch_size
        :return: None
        '''
        (m,n_H0,n_W0,n_C0) = X_train.shape
        n_y = y_train.shape[1]
        ##1.create placehold

        X,y = self.create_placehold(n_H0,n_W0,n_C0,n_y)
        ##2.initialize parameters

        parameters = self.initialize_parameters()
        ##3.forword propagation

        yHat = self.forword_propagation(X,parameters)
        ##4.compute cost
        cost = self.compute_cost(yHat,y)
        ##5.optimizer

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cost)

        ##6.initializer
        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            sess.run(init)
            costs = []
            for epoch in range(num_epochs):
                batchs = self.random_mini_batches(X_train,y_train,batch_size)
                batchs_cost = 0
                for batch in batchs:
                    (batch_X,batch_y) = batch
                    _,cos = sess.run([train,cost],feed_dict={X:batch_X,y:batch_y})
                    batchs_cost +=cos
                costs.append(batchs_cost/len(batchs))
                if epoch%10 == 0:
                    print('第'+str(epoch)+"次迭代的损失为",batchs_cost/len(batchs))
        return costs
if __name__ == '__main__':
    cnn = simple_CNN()
    x_path = 'F:/python/Deep Learing/data/cnn/train_signs.h5'
    y_path = 'F:/python/Deep Learing/data/cnn/test_signs.h5'
    X_train,y_train,X_test,y_test = cnn.load_data(x_path,y_path)
    cnn.fit(X_train,y_train)









