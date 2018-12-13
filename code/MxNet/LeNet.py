import mxnet as mx
from mxnet.gluon import loss as gloss,nn
from mxnet import autograd,init,gluon
import gluonbook as gb
import time
def create_model():
    #1.创建模型
    LeNet = nn.Sequential()
     #卷积
    LeNet.add(nn.Conv2D(6,kernel_size=5,strides=1,activation='sigmoid'),
              nn.MaxPool2D(pool_size=(2,2),strides=(2,2)),
              nn.Conv2D(16,kernel_size=5,strides=1,activation='sigmoid'),
              nn.MaxPool2D(pool_size=(2,2),strides=(2,2)))
     #全连接
    LeNet.add(nn.Dense(120,activation='sigmoid'),
              nn.Dense(84,activation='sigmoid'),
              nn.Dense(10))
    return LeNet
#3.训练模型(gpu训练)
def get_gpu():
    try:
        ctx = mx.gpu()
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
#评估函数
def evaluate_accuracy(data_iter,model,ctx):
    acc_sum = 0
    for x,y in data_iter:
        x,y = x.as_in_context(ctx),y.as_in_context(ctx)
        acc_sum +=gb.accuracy(model(x),y).mean()
    return acc_sum/len(data_iter)

def train(model,train_iter,test_iter,loss,Trainer,batch_size,num_epochs,ctx):
    '''
    :param model: 模型
    :param train_iter: train data
    :param test_iter: test data
    :param loss: loss function
    :param Trainer: trainer
    :param num_epochs: the times of iteration
    :param ctx: gpu or cpu
    :return: None
    '''
    print('train is on',ctx)
    start = time.time()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,epo_start = 0,0,time.time()
        for x,y in train_iter:
            x,y = x.as_in_context(ctx),y.as_in_context(ctx)
            with autograd.record():
                yhat = model(x)
                l = loss(yhat,y)
            l.backward()
            Trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += gb.accuracy(yhat,y)
        test_acc_sum = evaluate_accuracy(test_iter,model,ctx)
        print('epoch is %d,loss is %.4f,train acc is %.4f,test acc is %.4f,time is %.1f'%(epoch+1,train_l_sum/len(train_iter),
              train_acc_sum/len(train_iter),test_acc_sum,time.time()-epo_start))
    print('The sum of time is %.f'%(time.time()-start))
def train_model():
    #1.加载数据
    batch_size = 128
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

    #2.创建模型
    ctx = get_gpu()

    LeNet = create_model()
    LeNet.initialize(init=init.Xavier(),force_reinit=True,ctx=ctx)

    learning_rate = 0.9
    num_epochs = 5

    loss = gloss.SoftmaxCrossEntropyLoss()  ##损失函数
    Trainer = gluon.Trainer(LeNet.collect_params(),'sgd',{'learning_rate':learning_rate})
    train(LeNet,train_iter,test_iter,loss,Trainer,batch_size,num_epochs,ctx)

if __name__ == '__main__':
    train_model()