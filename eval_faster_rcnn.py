import os
import pprint
import moxing.mxnet as mox
import mxnet as mx
from moxing.mxnet.module.rcnn.tester import Predictor, pred_eval
from moxing.mxnet.data.rcnn_load.loader import TestLoader
from moxing.mxnet.data.rcnn_load.dataset.pascal_voc import PascalVOC
import argparse

def add_parameter():
    parser = argparse.ArgumentParser(description='train faster rcnn', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_url', type=str, default='ckpt/e2e-rcnn-resnet101', help='prefix of trained model file')
    parser.add_argument('--load_epoch', type=int, help='load the model on epoch use checkpoint_url')
    parser.add_argument('--cache_path', type=str, default='s3://obs-lpf/data/')
    args = parser.parse_args()
    return args

def e2e_eval():
    args = add_parameter()
    assert args.checkpoint_url != None, 'checkpoint_url should not be None'
    assert args.load_epoch != None, 'load_epoch should not be None'
    # set environment parameters
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    mox.rcnn_config.generate_config('resnet_rcnn', 'PascalVOC')
    mox.rcnn_config.config.TEST.HAS_RPN = True
    pprint.pprint(mox.rcnn_config.config)
    mox.rcnn_config.default.root_path = args.cache_path
    mox.rcnn_config.default.dataset_path = mox.get_hyper_parameter('data_url')
    # load data
    test_image_set = mox.rcnn_config.default.test_image_set
    root_path = mox.rcnn_config.default.root_path
    dataset_path = mox.rcnn_config.default.dataset_path
    imdb = PascalVOC(test_image_set, root_path, dataset_path)
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, batch_size=1, shuffle=False, has_rpn=True)
    # load symbol
    symbol = mox.get_model('object_detection', 'resnet_rcnn', num_classes=21, is_train=False)
    # load model params
    model_prefix = args.checkpoint_url
    load_epoch = args.load_epoch
    arg_params, aux_params = mox.rcnn_load_param(
        model_prefix, load_epoch,
        convert=True, data=test_data, process=True,
        is_train=False, sym=symbol)
    max_data_shape = [('data', (1, 3, max([v[0] for v in mox.rcnn_config.config.SCALES]), \
        max([v[1] for v in mox.rcnn_config.config.SCALES])))]
    # create predictor
    devs = [mx.gpu(0)]
    predictor = Predictor(
        symbol,
        data_names=[k[0] for k in test_data.provide_data],
        label_names=None,
        context=devs,
        max_data_shapes=max_data_shape,
        provide_data=test_data.provide_data,
        provide_label=test_data.provide_label,
        arg_params=arg_params,
        aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, vis=False, thresh=0.001)

if __name__ == '__main__':
    mox.file.set_auth(path_style=True)
    e2e_eval()
