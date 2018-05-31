import moxing.mxnet as mox
import time
import os
import pdb
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '48'
#os.environ['MXNET_CPU_PRIORITY_NTHREADS'] = '4'
#pdb.set_trace()
mox.set_hyper_parameter('data_url', 's3://obs-lpf/data/')
#mox.set_hyper_parameter('data_url', '/home/hzz/lipengfei/flower/')
#mox.set_hyper_parameter('train_file', 'flower_256_q100_train.rec')
#mox.set_hyper_parameter('val_file', 'flower_256_q100_test.rec')
mox.set_hyper_parameter('train_file', 'flower_raw')
mox.set_hyper_parameter('val_file', 'flower_raw')
(train_data, val_data) = mox.get_data_iter('imageraw', hyper_train={'data_shape': (3, 224, 224),'batch_size': 512, 'inter_method': 2}, num_process=128)
#(train_data, val_data) = mox.get_data_iter('imageraw', hyper_train={'data_shape': (3, 224, 224),'batch_size': 512, 'inter_method': 2})
#(train_data, val_data) = mox.get_data_iter('imagerecord', hyper_train={'data_shape': (3, 224, 224),'batch_size': 512,'preprocess_threads':48})
begin = time.time()
for i in range(10):
    tic = time.time()
    tmp = train_data.next()
    res = time.time() - tic
    #pdb.set_trace()
    print ("used time: %f"%res)
end = time.time() - begin
print ("avg time : %f"%(end/10))
