import numpy as np
def updata_param_with_sgd(parames,grades,learn_rate):
    '''
    使用SGD更新参数
    :param parames: W,b
    :param grades: dW,db
    :param learn_rate: alpha
    :return:
    '''
    length = len(parames)//2
    for i in range(1,length+1):
        parames['W'+str(i)] -= learn_rate*grades['dW'+str(i)]
        parames['b'+str(i)] -= learn_rate*grades['db'+str(i)]
    return parames

def updata_param_with_mini_sgd(X,y,parames,grades,learn_rate,batch_size):
    '''
    使用mini_sgd更新参数
    :param X :
    :param y :
    :param parames: W,b
    :param grades: dW,db
    :param learn_rate: alpha
    :param batch_size: batch的大小
    :return:
    '''
    data = np.hstack((X,y))
    length = len(parames)//2
    #将数据随机打乱
    data = np.random.shuffle(data)

    ##将数据分按batch_size的大小划分数据为多个batch
    batch_n = len(X)//batch_size
    batchs = []
    for batch in range(1,batch_n+2):
        start = batch_size*(batch-1)
        end = min(batch_size*batch_size)
        batch_x = data[start:end,:-1]
        batch_y = data[start:end,:-1]
        batchs.append((batch_x,batch_y))
    return batchs
    ##一轮的迭代
    for batch in range(len(batchs)):
        for i in range(len(length)):
            ##前向传播
            ##反向传播
            ##参数更新

