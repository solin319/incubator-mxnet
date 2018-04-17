import moxing.mxnet as mox
import mxnet as mx
import argparse
from moxing.mxnet.data.camvid_dataset import FileIter

def add_parameters():
    parser = argparse.ArgumentParser(description='eval segnet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_url', type=str, help='checkpoint path, path which provide model to read'
                                                           'it include model prefix')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--disp_batchs', type=int, default=10)
    parser.add_argument('--load_epoch', type=int)
    parser.add_argument('--test_file_name', type=str, default='test.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = add_parameters()
    mox.file.set_auth(path_style=True)
    data = FileIter(
        batch_size=args.batch_size,
        root_dir=mox.get_hyper_parameter('data_url'),
        flist_name=args.test_file_name,
        data_shape=(3, 360, 480),
        rgb_mean=[123.68, 116.779, 103.939],
        shuffle=False)

    #get symbol and parameters
    assert args.checkpoint_url != None, 'checkpoint_url should not be None'
    assert args.load_epoch != None, 'load_epoch should not be None'
    sym, arg_params, aux_params = mox.load_model(args.checkpoint_url, args.load_epoch)
    num_gpus = mox.get_hyper_parameter('num_gpus')
    devs = mx.cpu() if num_gpus is None else [
        mx.gpu(int(i)) for i in range(num_gpus)]
    model = mx.mod.Module(
        context       = devs,
        symbol        = sym,
        label_names   = ['softmax_label',]
    )
    params_tuples = (arg_params, aux_params)
    metrics = [mox.utils.contrib_metrics.Accuracy(ignore_label=11),
                      mox.utils.contrib_metrics.CrossEntropy()]
    batch_end_callback = mx.callback.Speedometer(args.batch_size, args.disp_batchs, auto_reset=False)
    mox.run(data, model, params_tuples,run_mode=mox.ModeKeys.EVAL, metrics=metrics, batch_end_callbacks=batch_end_callback)