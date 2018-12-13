import mxnet as mx
from mxnet import nd
b = nd.random.uniform(shape=(2,3),ctx=mx.gpu(0))
print(b)