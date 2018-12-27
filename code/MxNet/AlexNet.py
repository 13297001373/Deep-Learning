import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss ,nn
import gluonbook as gb
import time
'''
特征是可以学习的
'''
def create_model():
    AlexNet = nn.Sequential()

    AlexNet.add(nn.Conv2D(96,kernel_size=11,strides=4,activation='relu'), #第一层
                nn.MaxPool2D(pool_size=3,strides=2),

                nn.Conv2D(256,kernel_size=5,padding=2,activation='relu'),#第二层
                nn.MaxPool2D(pool_size=3,strides=2),

                nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),#第三层

                nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),#第四层

                nn.Conv2D(256,kernel_size=3,padding=1,activation='relu'),#第五层
                nn.MaxPool2D(pool_size=3,strides=2))
    AlexNet.add(nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
                nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
                nn.Dense(10))
    return AlexNet
def get_ctx():
    try:
        ctx = mx.gpu()
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
def load_data(batch_size):
    train_iter,test_iter = gb.load_data_fashion_mnist(batch_size,resize=224)
    return train_iter,test_iter

def accuracy(yhat,y):
    return (nd.argmax(yhat,axis=1)==y.astype('float32')).mean().asscalar()
def evaluate_accuracy(model,data_iter,ctx):
    acc = 0
    for x,y in data_iter:
        x,y = x.as_in_context(ctx),y.as_in_context(ctx)
        acc+=accuracy(model(x),y)
    return acc/len(data_iter)

def train(model,train_iter,test_iter,loss,Trainer,batch_size,num_epochs,ctx):
    print('train is on ',ctx)
    start = time.time()

    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,epoch_time = 0,0,time.time()
        for x,y in train_iter:
            x,y = x.as_in_context(ctx),y.as_in_context(ctx)
            with autograd.record():
                yhat = model(x)
                l = loss(yhat,y)
            l.backward()
            Trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(yhat,y)
        test_acc_sum = evaluate_accuracy(model,test_iter,ctx)
        print('epoch is %d , loss in %.f,train acc is %.f,test acc is %.f,time is %.f'%(epoch+1,train_l_sum/len(train_iter),
                                                                            train_acc_sum/len(train_iter),
                                                                            test_acc_sum,time.time()-epoch_time))
    print(time.time()-start)
def train_model():
    #$加载数据
    batch_size =64
    train_iter,test_iter = load_data(batch_size)

    ##获取模型
    ctx = get_ctx()
    AlexNet = create_model()
    AlexNet.initialize(force_reinit=True,init=init.Xavier(),ctx=ctx)
    learning_rate = 0.1
    num_epochs = 5
    loss = gloss.SoftmaxCrossEntropyLoss()
    Trainer = gluon.Trainer(AlexNet.collect_params(),'sgd',{'learning_rate':learning_rate})
    train(AlexNet,train_iter,test_iter,loss,Trainer,batch_size,num_epochs,ctx)

if __name__ == '__main__':
    train_model()
    # model = create_model()
    # X = nd.random.uniform(shape=(1, 1, 224, 224))
    # model.initialize()
    # for layer in model:
    #     X = layer(X)
    #     print(layer.name,X.shape)