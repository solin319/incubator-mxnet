# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""`MXNetBaseService` defines an API for MXNet service.
"""
import base64
import mxnet as mx
import requests
import zipfile
import json
import shutil
import os
import numpy as np
import logging

from mxnet.io import DataBatch
from mms.model_service.mxnet_model_service import MXNetBaseService
from mms.utils.mxnet import image, ndarray

SHORTER_SIZE = 600
MAX_SIZE = 1000
IMAGE_STRIDE = 0
NUM_CLASSES = 21
THRESH = 0.001
TEST_NMS = 0.3

def rcnn_resize(im, target_size, max_size, stride=0):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = image.resize(im,
                      int(im.shape[1] * float(im_scale)),
                      int(im.shape[0] * float(im_scale)),
                      interp=1)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def nonlinear_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes

def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class DLSFasterRCNNService(MXNetBaseService):
    '''MXNetBaseService defines the fundamental loading model and inference
       operations when serving MXNet model. This is a base class and needs to be
       inherited.
    '''
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        self.model_name = model_name
        self.ctx = mx.gpu(int(gpu)) if gpu is not None else mx.cpu()
        self._signature = manifest['Model']['Signature']
        self.data_names = []
        self.data_shapes = []
        self.scale = 1
        for input in self._signature['inputs']:
            self.data_names.append(input['data_name'])
            # Replace 0 entry in data shape with 1 for binding executor.
            # Set batch size as 1
            self.data_shape = input['data_shape']
            self.data_shape[0] = 1
            for idx in range(len(self.data_shape)):
                if self.data_shape[idx] == 0:
                    self.data_shape[idx] = 1
            self.data_shapes.append((input['data_name'], tuple(self.data_shape)))

        # add im_info to input name and shapes
        self.data_names.append('im_info')
        self.data_shapes.append(('im_info', (1,3)))

        # Load MXNet module
        epoch = 0
        try:
            param_filename = manifest['Model']['Parameters']
            epoch = int(param_filename[len(model_name) + 1: -len('.params')])
        except Exception as e:
            logging.info('Failed to parse epoch from param file, setting epoch to 0')

        sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (model_dir, manifest['Model']['Symbol'][:-12]), epoch)
        ## process the arg_params: remove the '_test' in tail of names
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
        ## process the arg_params
        self.symbol = sym
        self.mx_model = mx.mod.Module(symbol=sym, context=self.ctx,
                                      data_names=self.data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=self.data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            img_arr = image.read(img)
            img_arr, im_scale = rcnn_resize(img_arr,
                                            SHORTER_SIZE,
                                            MAX_SIZE,
                                            stride=IMAGE_STRIDE)
            rgb_mean = mx.nd.array([[[0, 0, 0]]])
            img_arr = img_arr.astype('float32')
            img_arr = img_arr - rgb_mean
            img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)
            im_info = [[img_arr.shape[2], img_arr.shape[3], im_scale]]
            self.scale = im_scale
            img_list.append(mx.nd.array(im_info))
        return img_list

    def _postprocess(self, data):
        output = dict(zip(self.mx_model.output_names, data))
        rois = output['rois_output'].asnumpy()[:, 1:]

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

        # post processing
        pred_boxes = nonlinear_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, self.mx_model.data_shapes[0].shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        boxes = pred_boxes / self.scale
        all_boxes = [[] for _ in xrange(NUM_CLASSES)]
        for j in range(1, NUM_CLASSES):
            indexes = np.where(scores[:, j] > THRESH)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets, TEST_NMS)
            all_boxes[j] = cls_dets[keep, :]
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, NUM_CLASSES)]
        response = []
        for i, item in enumerate(boxes_this_image):
            if len(item) > 0:
                for sub_item in item:
                    response.append((i, sub_item[4], sub_item[0],
                                     sub_item[1], sub_item[2], sub_item[3]))
        return response

    def _inference(self, data):
        '''Internal inference methods for MXNet. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        '''
        # Check input shape
        data = [item.as_in_context(self.ctx) for item in data]
        input_shapes = [('data', data[0].shape), ('im_info', data[1].shape)]
        shape_changed = False
        for i, item in enumerate(input_shapes):
            if item != self.data_shapes[i]:
                shape_changed = True
                break
        # rebind for new shape
        if shape_changed:
            self.data_shapes = input_shapes
            new_module = mx.mod.Module(self.symbol, context=self.ctx,
                                       data_names=['data', 'im_info'], label_names=None)
            new_module.bind(data_shapes=input_shapes,
                            for_training=False,
                            force_rebind=False,
                            shared_module=self.mx_model)
            self.mx_model = new_module
        self.mx_model.forward(DataBatch(data))
        return self.mx_model.get_outputs()

    def ping(self):
        '''Ping to get system's health.

        Returns
        -------
        String
            MXNet version to show system is healthy.
        '''
        return mx.__version__

    @property
    def signature(self):
        '''Signiture for model service.

        Returns
        -------
        Dict
            Model service signiture.
        '''
        return self._signature
