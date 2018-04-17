import mxnet as mx
import moxing.mxnet as mox
import argparse

def add_parameters():
    parser = argparse.ArgumentParser(description='train segmentation', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_classes', type=int, default=11,
                        help='the number of classes')
    parser.add_argument('--network', type=str, default='segnet', help='network name')
    parser.add_argument('--lr', type=float, default=0.01, help='initail learning rate')
    parser.add_argument('--batch_size', type=int, default=12, help='number of examples per batch')
    parser.add_argument('--wd', type=float, default=0.001, help='weight decay for sgd')
    parser.add_argument('--load_epoch', type=int,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--checkpoint_url', type=str, default='s3://obs-lpf/ckpt/segnet/segnet',
                        help='checkpoint path, path which provide model to read' \
                             'it include model prefix')
    parser.add_argument('--vgg_pretrained_url', type=str,
                        help='vgg pretrained model prefix')
    parser.add_argument('--save_frequency', type=int, default=5,
                        help='number of epochs to save network')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='the number of training epochs')
    args = parser.parse_args()
    return args

def get_data(args):
    from moxing.mxnet.config.data_config import camvid_config
    camvid_config.data_path = mox.get_hyper_parameter('data_url')
    data_set = mox.get_data_iter('camvid',
                                 hyper_train={'batch_size':args.batch_size},
                                 hyper_val={'batch_size':args.batch_size})
    return data_set

def get_optimizer(args):
    optimizer_params = mox.get_optimizer_params(num_examples=360,
                                                lr=args.lr,
                                                batch_size=args.batch_size,
                                                wd=args.wd,
                                                lr_step_epochs='150, 200')
    return optimizer_params

def design_symbol():
    net = mox.get_model('segmentation', args.network, num_classes=args.num_classes)
    return net

def get_model():
    num_gpus = mox.get_hyper_parameter('num_gpus')
    devs = mx.cpu() if num_gpus is None else [mx.gpu(int(i)) for i in range(num_gpus)]
    model = mx.mod.Module(
        context       = devs,
        symbol        = design_symbol()
    )
    return model

if __name__ == '__main__':
    args = add_parameters()
    mox.file.set_auth(path_style=True)
    mox.set_hyper_parameter('disp_batchs', 5)
    mox.set_hyper_parameter('save_frequency', args.save_frequency)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    metrics = [mox.contrib_metrics.Accuracy(ignore_label=11),
               mox.contrib_metrics.CrossEntropy()]
    if args.load_epoch is not None:
        if args.vgg_pretrained_url is not None:
            sym, arg_params, aux_params = mox.load_model(args.vgg_pretrained_url, args.load_epoch)
            params_tuple = (arg_params, aux_params)
        elif args.checkpoint_url is not None:
            sym, arg_params, aux_params = mox.load_model(args.checkpoint_url, args.load_epoch)
            params_tuple = (arg_params, aux_params)
    else:
        params_tuple = (None, None)
    mox.run(data_set=get_data(args),
            optimizer='sgd',
            optimizer_params=get_optimizer(args),
            run_mode=mox.ModeKeys.TRAIN,
            model=get_model(),
            initializer=initializer,
            batch_size=args.batch_size,
            epoch_end_callbacks=mox.save_model(args.checkpoint_url),
            params_tuple=params_tuple,
            metrics=metrics,
            num_epochs=args.num_epochs,)
