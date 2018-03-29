import os
import argparse
import logging
from multiprocessing import cpu_count
import mxnet as mx

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)

def get_data(args):
    train_data_path = os.path.join(args.data_url, args.train_file)
    val_data_path = os.path.join(args.data_url, args.val_file)
    num_thread = max(cpu_count() / 2 - 1, 1)
    if train_data_path is not None:
        train = mx.io.ImageRecordIter(
            path_imgrec=train_data_path,
            path_imgidx='',
            label_with=1,
            mean_r=123.68,
            mean_g=116.779,
            mean_b=103.939,
            data_name='data',
            label_name='softmax_label',
            data_shape=(3, 224, 224),
            batch_size=args.batch_size,
            rand_crop=1,
            rand_mirror=1,
            preprocess_threads=num_thread,
            shuffle=1
        )
    if val_data_path is not None:
        val = mx.io.ImageRecordIter(
            path_imgrec=val_data_path,
            path_imgidx='',
            label_with=1,
            mean_r=123.68,
            mean_g=116.779,
            mean_b=103.939,
            data_name='data',
            label_name='softmax_label',
            batch_size=args.batch_size,
            data_shape=(3, 224, 224),
            preprocess_threads=num_thread
        )
    return (train, val)

def get_optimizer_params(args):
    optimizer_params = {'learning_rate': args.lr,
                        'wd': 0.0001,
                        'clip_gradient': 5,
                        'rescale_grad': (1.0 / (args.batch_size))}
    return optimizer_params

def get_model(network):
    num_gpus = args.num_gpus
    devs = mx.cpu() if num_gpus is None else [mx.gpu(int(i)) for i in range(num_gpus)]
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
    return mx.mod.Module(context=devs, symbol=network)

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_url', type=str, default='s3://lpf/data/flower/',
                        help='training record file to use')
    parser.add_argument('--train_file', type=str, default='flower_256_q100_train.rec',
                        help='file name of train dataset, such as ILSVRC2012_img_train.rec')
    parser.add_argument('--val_file', type=str, default='flower_256_q100_test.rec',
                        help='file name of train dataset, such as ILSVRC2012_img_val.rec')
    parser.add_argument('--num_gpus', type=int, help='number of gpus to use')
    parser.add_argument('--checkpoint_url', type=str, help='the pre-trained model')
    parser.add_argument('--layer_before_fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    parser.add_argument('--num_classes', type=int, help='the number of classes')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1, help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='the ratio to reduce lr on each step')
    parser.add_argument('--save_frequency', type=int, default=1,
                        help='how many epochs to save model')
    parser.add_argument('--kv_store', type=str, default='device',
                        help='key-value store type')
    parser.add_argument('--disp_batches', type=int, default=10,
                        help='show progress for every n batches')
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG)
    # load data
    data_set = get_data(args)

    # load pretrained model
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.checkpoint_url, args.load_epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)

    #load modle
    model = get_model(new_sym)
    params_tuple = (new_args, aux_params)

    kv = mx.kvstore.create(args.kv_store)
    worker_id = kv.rank
    save_path = args.checkpoint_url if worker_id == 0 else "%s-%d" % (args.checkpoint_url, worker_id)
    epoch_end_callbacks = mx.callback.do_checkpoint(save_path, args.save_frequency)
    metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]
    # train
    model.fit(train_data=data_set[0],
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_data=data_set[1],
              eval_metric=metrics,
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=get_optimizer_params(args),
              initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
              arg_params=new_args,
              aux_params=aux_params,
              batch_end_callback=[mx.callback.Speedometer(args.batch_size, args.disp_batches)],
              epoch_end_callback=epoch_end_callbacks,
              allow_missing=True)
