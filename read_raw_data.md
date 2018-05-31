## 读原始图片

### 数据集准备

原始图片，需要按如下目录形式组织：

![](./pic/dir.png)

### 接口说明

#### 接口

沿用mox.get_data_iter接口，添加'imageraw'模式，当传入num_process参数时，使用多进程读数据模块，否则单进程读数据

```
(train_data, val_data) = mox.get_data_iter('imageraw', hyper_train={'data_shape': (3, 224, 224),'batch_size': 512, 'inter_method': 2}, num_process=128)
```

#### 参数设置

data_url设置为数据集总目录data_path，mox.set_hyper_parameter('data_url', 'data_path')

train_file设置为训练集子目录名train_data_path，mox.set_hyper_parameter('train_file', 'train_data_path')

val_file设置为验证集子目录名train_data_path，mox.set_hyper_parameter('train_file', 'val_data_path')

与单进程模块相比新增4个参数：

num_process：进程数，默认128

capacity：缓存队列长度，默认1024

shared_array_size：队列中每个缓存模块的大小，默认5120

ctx：batch_data读取后存储的设备位置，默认cpu

#### 实现

实现类RawImageIterAsync继承于mx.io.DataIter

![](./pic/func.png)

子进程执行逻辑

![](./pic/sub_process.png)

主进程next函数逻辑

![](./pic/next.png)

### 验证

测试读取数据时间，具体操作包括：read+decode+resize+crop+transpose，单位秒

batch_size=512，MXNET_CPU_WORKER_NTHREADS=48

结果如下：

|              | 裸机-本地    | 裸机-obs   | 类生产-obs   | 云道-本地    | 云道-obs   |
| ------------ | -------- | -------- | --------- | -------- | -------- |
| rec文件        | 0.061895 | 0.609195 | 0.300987  | 0.087625 | 0.054941 |
| jpg文件-单进程    | 0.352382 | 43.8396  | 35.212893 | 0.330239 | 2.745249 |
| jpg文件-128多进程 | 0.289777 | 0.336491 | 0.331482  | 0.307799 | 0.315927 |

云道裸金属，resnet-50测试，训练flower数据集，kv_store=device，batch_size_per_gpu=64，单位samples/sec

|      | 虚拟数据 | raw-单进程-本地 | raw-多进程-本地 | raw-多进程-obs | rec-obs |
| ---- | ---- | ---------- | ---------- | ----------- | ------- |
| 单卡   | 195  | 194        | 194        | 194         | 194     |
| 8卡   | 1526 | 1407       | 1450       | 1410        | 1523    |

### 总结

1. 云道单机8卡节点，cpu多进程处理能力受限，导致性能瓶颈，待验证。
2. 子进程目前只负责读取数据流，设计子进程继续负责图片解码、预处理，将mx.NDArray内存地址放入共享内存传给主进程，再由主进程通过内存地址重建mx.NDArray。这个方案会带来近乎1倍的性能提升（本地裸机测试），但存在一定风险：分配的进程数量增加，在容器中执行导致性能下降，可能是启动容器时做出的cpu线程数限制导致，待验证；子进程创建的内存空间是否会在主进程使用前被回收，待验证。
3. 当batch_size较大时，不仅读数据会产生瓶颈，输入数据的拷贝cpu-gpu也会导致性能下降，所以在多进程读模块中设计了ctx参数，将读入的数据直接放入指定设备，放入gpu中可以缓解上述瓶颈，将多进程读的整体性能提升3%。
4. 经测试将batch_data存入cpu_pinned，经测试速度与放入gpu相似，均可缓解cpu-gpu拷贝的瓶颈。
5. 多次python调用c++，可能导致性能下降，使用cpu_pinned存储，将显式的uint8->float32转换调用，变为内存拷贝过程中隐式的类型转换，可以将裸机本地读原始图片的时间从0.289s降低到0.238。