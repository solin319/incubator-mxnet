import mxnet as mx
import argparse
import logging
from resnet import get_symbol

def get_mnist_iter(args):
    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_url + 'cifar10_train.rec',
        path_imgidx='',
        label_width=1,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        data_name='data',
        label_name='softmax_label',
        data_shape=(3, 28, 28),
        batch_size=args.batch_size,
        pad=4,
        fill_value=127,
        rand_mirror=1,
        rand_crop=1,
        preprocess_threads=4,
        shuffle=True)

    return train

def fit(args):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    train = get_mnist_iter(args)
    checkpoint = mx.callback.do_checkpoint(args.model_prefix if kv.rank == 0 else "%s-%d" % (
        args.model_prefix, kv.rank))
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    network = get_symbol(num_classes=args.num_classes, num_layers=110, image_shape='3, 28, 28')
    devs = mx.cpu() if args.num_gpus == 0 else [mx.gpu(int(i)) for i in range(args.num_gpus)]
    # create model
    model = mx.mod.Module(context=devs, symbol=network)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    optimizer_params = {'learning_rate': args.lr, 'wd' : 0.0001}
    # run
    model.fit(train,
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_data=None,
              eval_metric=['accuracy'],
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=None,
              aux_params=None,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=True)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num_examples', type=int, default=50000,
                        help='the number of training examples')

    parser.add_argument('--data_url', type=str, default='s3://obs-lpf/data/cifar10/', help='the training data')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='max num of epochs')
    parser.add_argument('--disp_batches', type=int, default=5,
                        help='show progress for every n batches')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--kv_store', type=str, default='device',
                        help='key-value store type')
    parser.add_argument('--model_prefix', type=str, default='s3://obs-lpf/ckpt/cifar/cifar10_resnet_110',
                        help='model prefix')
    parser.add_argument('--num_gpus', type=int, default='0',
                        help='number of gpus')
    args, unkown = parser.parse_known_args()

    fit(args)
