import numpy as np

def relu(X):
    return np.max(X,0)
class NN_Drop(object):
    def __init__(self):
        self.params = {}

    def init_params(self, layer_dims, types):
        '''
        :param layer_dims: 各层的维度，type：list
        :param types: 初始化方式（'zeros','random','he'）,type:str
        :return: 各层变量初始化
        '''
        params = {}
        if types == 'zeros':  ##0初始化
            for i in range(len(layer_dims) - 1):
                params['W' + str(i + 1)] = np.zeros(shape=(layer_dims[i + 1], layer_dims[i]))
                params['b' + str(i + 1)] = np.zeros(shape=(layer_dims[i + 1], 1))
        elif types == 'random':
            for i in range(len(layer_dims) - 1):
                params['W' + str(i + 1)] = np.random.rand(layer_dims[i + 1], layer_dims[i])
                params['b' + str(i + 1)] = np.random.rand(layer_dims[i + 1], 1)
        elif types == 'he':  ##防止出现梯度爆炸或者梯度消失
            for i in range(len(layer_dims) - 1):
                params['W' + str(i + 1)] = np.random.rand(layer_dims[i + 1], layer_dims[i]) / np.sqrt(2 / layer_dims[i])
                params['b' + str(i + 1)] = np.random.rand(layer_dims[i + 1], 1) / np.sqrt(2 / layer_dims[i])
        else:
            print('type 参数错误！！')
        return params
    def forward_propagetion_with_drop_out(X,keep_prob=0.5):
        '''
        forward_propagation
        :param keep_prob: droup_out prob
        :return: result,cache
        '''
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        W3 = self.params['W3']
        b3 = self.params['b3']
        ##Linear->relu->linear->relu->linear->sigmoid
        ##step-1
        Z1 = np.dot(W1,X)
        A1 = relu(Z1)
        D1 = np.random.rand(A1.shape[0],A1.shape[1])

        ##drop_out
        D1 = D1<keep_prob
        A1 = np.multiply(A1,D1)
        A1 = A1 / keep_prob

        Z2 = np.dot(W2,A1)
        A2 = relu(Z2)
