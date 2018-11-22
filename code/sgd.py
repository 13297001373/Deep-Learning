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
            pass
def update_parameters_with_momentum(parames,grades,v,beta,learn_rate):
    '''
    使用momentum更新参数
    :param parames: W,b
    :param grades: dW,db
    :param grades: v
    :param grades: beta
    :param learn_rate: alpha
    :return:
    '''
    length = len(parames)//2
    for i in range(1,length+1):
        ##init v
        v['dW'+str(i)] = beta*v['dW'+str(i)] + (1-beta)*grades['dW'+str(i)]
        v['db'+str(i)] = beta*v['db'+str(i)] + (1-beta)*grades['db'+str(i)]

        ##init paramers
        parames[W+str(i)] -=learn_rate*v['dW'+str(i)]
        parames[b+str(i)] -=learn_rate*v['db'+str(i)]
    return parames,v

def update_parameters_with_Adam(parames,grades,v,s,beta1,beta2,t,learn_rate,epsilon):
    '''
    使用Adam更新参数
    :param parames: W,b
    :param grades: dW,db
    :param v: v
    :param s: s
    :param beta1:
    :param beta2:
    :param learn_rate:
    :param epsilon:
    :return:
    '''
    length = len(parames)//2
    for i in range(1,length+1):
        ## 第一次更新v
        v['dW'+str(i)] = beta1*v['dW'+str(i)] + (1-beta1)*v['dW'+str(i)]
        v['db'+str(i)] = beta1*v['db'+str(i)] + (1-beta1)*v['db'+str(i)]

        ##修正v
        v['dW'+str(i)] = v['dW'+str(i)]/(1-np.power(beta1,t))
        v['db'+str(i)] = v['db'+str(i)]/(1-np.power(beta1,t))

        ##第一次更新s
        s['dW'+str(i)] = beta2*s['dW'+str(i)] + (1-beta2)*s['dW'+str(i)]
        s['db'+str(i)] = beta2*s['db'+str(i)] + (1-beta2)*s['dW'+str(i)]

        ##修正s
        s['dW'+str(i)] = s['dW'+str(i)]/(1-np.power(beta2,t))
        s['db'+str(i)] = s['db'+str(i)]/(1-np.power(beta2,t))

        ##更新 parames
        parames['W'+str(i)] -= learn_rate*v['dW'+str(i)]/(np.sqrt(s['dW'+str(i)])+epsilon)
        parames['b'+str(i)] -= learn_rate*v['db'+str(i)]/(np.sqrt(s['db'+str(i)])+epsilon)
    return parames

