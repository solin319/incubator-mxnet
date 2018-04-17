import mxnet as mx
import moxing.mxnet as mox
import logging
import os
import numpy as np
import argparse

def add_parameter():
    parser = argparse.ArgumentParser(description='train faster rcnn', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet_rcnn', help='name of network')
    parser.add_argument('--dataset', type=str, default='PascalVOC', help='dataset name, defaultly use PascalVOC' )
    parser.add_argument('--checkpoint_url', type=str, default='ckpt/e2e-rcnn-resnet101', help='prefix of trained model file')
    parser.add_argument('--load_epoch', type=int, help='load the model on epoch use checkpoint_url')
    parser.add_argument('--pretrained', type=str, default='ckpt/resnet-101', help='pretrained model prefix')
    parser.add_argument('--pretrained_epoch', type=int, default=0, help='pretrained model epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--cache_path', type=str, default='s3://obs-lpf/data/')
    args = parser.parse_args()
    return args

def get_model(data_set,symbol):
    input_batch_size = data_set[0].batch_size
    max_data_shape = [('data', (input_batch_size, 3, max([v[0] for v in mox.rcnn_config.config.SCALES]), \
                                max([v[1] for v in mox.rcnn_config.config.SCALES])))]
    max_data_shape, max_label_shape = data_set[0].infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (input_batch_size, 100, 5)))
    logger = logging.getLogger()
    logger.info('providing maximum shape %s %s' % (max_data_shape, max_label_shape))
    num_gpus = mox.get_hyper_parameter('num_gpus')
    devs = mx.cpu() if num_gpus is None else [mx.gpu(int(i)) for i in range(num_gpus)]
    model = mox.MutableModule(
        symbol,
        data_names=[k[0] for k in data_set[0].provide_data],
        label_names=[k[0] for k in data_set[0].provide_label],
        logger=logger, context=devs, work_load_list=None,
        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
        fixed_param_prefix=mox.rcnn_config.config.FIXED_PARAMS)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    return model

if __name__ == '__main__':
    args = add_parameter()
    mox.file.set_auth(path_style=True)
    mox.rcnn_config.generate_config(args.network, args.dataset)
    mox.rcnn_config.default.image_set = '2007_trainval'
    mox.rcnn_config.default.root_path = args.cache_path
    mox.rcnn_config.default.dataset_path = mox.get_hyper_parameter('data_url')
    # training
    symbol = mox.get_model('object_detection', 'resnet_rcnn', num_classes=21)

    data_set = mox.data.get_data_iter('pascalvoc', sym=symbol)
    num_examples = len(data_set[0].roidb)
    batch_size = data_set[0].batch_size
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    metrics = [mox.contrib_metrics.RPNAccMetric(),
               mox.contrib_metrics.RPNLogLossMetric(),
               mox.contrib_metrics.RPNL1LossMetric(),
               mox.contrib_metrics.RCNNAccMetric(),
               mox.contrib_metrics.RCNNLogLossMetric(),
               mox.contrib_metrics.RCNNL1LossMetric(),
               mx.metric.CompositeEvalMetric()]

    num_classes = mox.rcnn_config.config.NUM_CLASSES
    if 'rfcn' in args.network:
        means = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_MEANS), num_classes * 7 * 7)
        stds = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_STDS), num_classes * 7 * 7)
    else:
        means = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_MEANS), num_classes)
        stds = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_STDS), num_classes)
    epoch_end_callbacks = mox.rcnn_do_checkpoint(args.checkpoint_url, means, stds)

    prefix = args.checkpoint_url if args.resume else args.pretrained
    epoch = args.load_epoch if args.resume else args.pretrained_epoch
    params_tuple = mox.rcnn_load_param(prefix, epoch, convert=True, \
                                             data=data_set[0], sym=symbol)

    lr=args.lr
    lr_factor = 0.1
    lr_iters = [int(epoch * num_examples / batch_size) for epoch in [3]]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    mox.run(data_set=(data_set[0], None),
            optimizer='sgd',
            optimizer_params=optimizer_params,
            run_mode=mox.ModeKeys.TRAIN,
            model=get_model(data_set, symbol),
            epoch_end_callbacks=epoch_end_callbacks,
            initializer=initializer,
            batch_size=batch_size,
            params_tuple=params_tuple,
            metrics=metrics,
            num_epochs=5)
