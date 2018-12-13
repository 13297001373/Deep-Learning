import mxnet as mx
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import loss as gloss ,nn
import gluonbook as gb
import time

def create_model():
    AlexNet = nn.Sequential()

    AlexNet.add(nn.Conv2D(96,kernel_size=11,strides=4,activation='relu'), #第一层
                nn.MaxPool2D(pool_size=3,strides=2),

                nn.Conv2D(256,kernel_size=5,padding=2,activation='relu'),#第二层
                nn.MaxPool2D(pool_size=3,strides=2),

                nn.Conv2D(384,kernel_size=3,strides=1,padding=1,activation='relu'),#第三层

                nn.Conv2D(384,kernel_size=3,strides=1,padding=1,activation='relu'),#第四层

                nn.Conv2D(256,kernel_size=3,strides=1,padding=1,activation='relu'),#第五层
                nn.MaxPool2D(pool_size=3,strides=2))
    AlexNet.add(nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
                nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
                nn.Dense(10))
    return AlexNet

if __name__ == '__main__':
    model = create_model()
    X = nd.random.uniform(shape=(1,1,224,224))
    model.initialize()
    for layer in model:
        X = layer(X)
        print(layer.name,X.shape)