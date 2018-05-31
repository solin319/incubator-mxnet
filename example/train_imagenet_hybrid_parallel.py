import mxnet as mx
import os
import moxing.mxnet as mox
import argparse
from importlib import import_module
from moxing.mxnet.symbols.classification.vgg import get_symbol
import logging
logging.basicConfig(level=logging.DEBUG)

def add_parameter():
    parser = argparse.ArgumentParser(description='train cifar', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_train', help='train file path', type=str,
                        default = 's3://obs-lpf/data/imagenet/ILSVRC2012_img_train.rec')
    parser.add_argument('--data_val', help='val file path', type=str,
                        default = 's3://obs-lpf/data/imagenet/ILSVRC2012_img_val.rec')
    parser.add_argument('--train_type', default='dp', type=str,
                        help='dp is traditional data parallel, hp is hybrid parallel')
    parser.add_argument('--gpus', type=int, default=4,
                        help='num gpus used to train')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='the number of classes')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='max num of epochs')
    parser.add_argument('--checkpoint_url', type=str,
                        help='checkpoint path, path which provide model to read' \
                             'it include model prefix')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--disp_batches', type=int, default=5,
                        help='show progress for every n batches')
    args, _ = parser.parse_known_args()
    return args

def get_data(args):
    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_train,
        path_imgidx='',
        label_width=1,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        data_name='data',
        label_name='softmax_label',
        data_shape=(3, 224, 224),
        batch_size=args.batch_size,
        rand_crop=True,
        fill_value=127,
        rand_mirror=True,
        preprocess_threads=16,
        shuffle=True,
        num_parts=1,
        part_index=0)
    val = mx.io.ImageRecordIter(
        path_imgrec=args.data_val,
        path_imgidx='',
        label_width=1,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        data_name='data',
        label_name='softmax_label',
        data_shape=(3, 224, 224),
        batch_size=args.batch_size,
        preprocess_threads=16,
        num_parts=1,
        part_index=0)
    return (train, val)

def train():
    args = add_parameter()
    (train, val) = get_data(args)
    network = get_symbol(num_classes=args.num_classes, num_layers=16, batch_norm=True)
    checkpoint = None
    if args.checkpoint_url is not None:
        checkpoint = mx.callback.do_checkpoint(args.checkpoint_url)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler([30*10000, 60*10000, 90*10000], 0.1)
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': 0.01,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / args.batch_size),
                        'clip_gradient': 5}
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in range(args.gpus)]
    model = mx.mod.Module(context=devs, symbol=network)
    eval_metrics = ['accuracy', 'ce']
    kv_store = mx.kvstore.create('device')
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    initializer=mx.init.Xavier()
    opt = 'sgd'
    if args.train_type == 'hp':
        mox.fit(fit_type='hybrid_parallel',
                model=model,
                data_set=(train, val),
                metrics=eval_metrics,
                kvstore=kv_store,
                optimizer=opt,
                optimizer_params=optimizer_params,
                initializer=initializer,
                arg_params=None,
                aux_params=None,
                batch_end_callbacks=batch_end_callbacks,
                epoch_end_callbacks=checkpoint,
                num_epochs=args.num_epochs,
                batch_fc_type=1,
                batch_fc_gpus=1,
                batch_fc_num=1,
                batch_fc_machines=1,
                fc_sync_step=1)
    elif args.train_type == 'dp':
        model.fit(train,
                  begin_epoch=0,
                  num_epoch=args.num_epochs,
                  eval_data=val,
                  eval_metric=eval_metrics,
                  kvstore=kv_store,
                  optimizer=opt,
                  optimizer_params=optimizer_params,
                  initializer=initializer,
                  arg_params=None,
                  aux_params=None,
                  batch_end_callback=batch_end_callbacks,
                  epoch_end_callback=checkpoint,
                  allow_missing=True)

if __name__ == '__main__':
    train()
