import tensorflow as tf

##生成文件队列
filelists = ['A.csv','B.csv','C.csv']
file_queue = tf.train.string_input_producer(filelists,shuffle=False)

##定义Reader
reader = tf.TextLineReader()
key,value = reader.read(file_queue)

##定义Decoder
example,label = tf.decode_csv(value,record_defaults=[['null'],['null']])
example_batch, label_batch = tf.train.shuffle_batch([example,label], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print(example_batch.eval(),label_batch.eval())
    coord.request_stop()
    coord.join(threads)
