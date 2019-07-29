#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:14:53 2019

@author: gaomingda
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import workspace, core

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.rpn_generator import generate_rpn_on_dataset
from detectron.core.rpn_generator import generate_rpn_on_range
from detectron.core.test import im_detect_all
from detectron.datasets import task_evaluation
from detectron.datasets.json_dataset import JsonDataset
from detectron.modeling import model_builder
from detectron.utils.io import save_object
from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.net as net_utils
import detectron.utils.subprocess as subprocess_utils
import detectron.utils.vis as vis_utils

import detectron.core.test_engine as test_engine
from detectron.core.config import merge_cfg_from_file
#import sys
#sys.path.append('/home/gaomingda/softer_nms_LIP_JPPNet/detectron/LIP_JPPNet')
#from evaluate_pose_JPPNet import generate_poseJPPNet_pred_model
import detectron.core.test_res_top as test_res_top

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pprint

from detectron.core.config import assert_and_infer_cfg

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)

root_path = os.path.dirname(os.path.realpath(__file__))

File = '/home/gaomingda/Documents/maskrcnn_fcn/demo/features/atr_image/997_15.jpg'
cfg_file =  '/home/gaomingda/Documents/maskrcnn_fcn/configs/getting_started/e2e_mask_rcnn_R-50-FPN_1x_init_ATR.yaml'
weights_file = '/home/gaomingda/Documents/maskrcnn_fcn/detectron-output/train/ATR_train/generalized_rcnn/3_2/model_final.pkl'
gpu_id = 0

merge_cfg_from_file(cfg_file)
cfg.TEST.WEIGHTS = weights_file
cfg.TRAIN.WEIGHTS = weights_file
assert_and_infer_cfg()
logger.info('Testing with config:')
logger.info(pprint.pformat(cfg))
    
model = test_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS, gpu_id=gpu_id)

im = cv2.imread(File)
entry = {}
entry['id'] = File.split('/')[-1].split('.')[0]
entry['image'] = File
print(entry)
timers = defaultdict(Timer)
with c2_utils.NamedCudaScope(gpu_id):
    cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
        model, im, None, timers, entry
    )
    
# fetch blob
# hg_pred_NCHW_8, fpn_inner_res3_3_sum_lateral, fpn_inner_res3_3_sum_lateral_pose_concat_conv
with c2_utils.NamedCudaScope(gpu_id):
    pose_map = workspace.FetchBlob(core.ScopedName('hg_pred_NCHW_16'))[0, :, :, :]
    res_feat = workspace.FetchBlob(core.ScopedName('fpn_inner_res4_5_sum_lateral'))[0, :, :, :]
    concat_conv_relu = workspace.FetchBlob(core.ScopedName('fpn_inner_res5_lateral_pose_conv'))[0, :, :, :]
    
blobs = [ pose_map, res_feat, concat_conv_relu]    
blobs_name = ['pose_map', 'res_feat', 'concat_conv_relu']
save_path = '/home/gaomingda/Documents/maskrcnn_fcn/demo/features/997_15'
#feat1 = pose_map[0, :, :]
plt.figure()
plt.imshow(np.sum(pose_map, axis=0)-pose_map[-1, :, :], cmap=cm.jet, interpolation='lanczos')
plt.axis('off')
plt.savefig(os.path.join(save_path, 'pose_sum25.png'))
plt.imshow(np.sum(pose_map, axis=0), cmap=cm.jet, interpolation='lanczos')
plt.axis('off')
plt.savefig(os.path.join(save_path, 'pose_sum26.png'))
# res_feat
plt.imshow(np.sum(res_feat, axis=0), cmap=cm.jet, interpolation='lanczos')
plt.axis('off')
plt.savefig(os.path.join(save_path, 'res_feat_sum.png'))
# concat_feat
plt.imshow(np.sum(concat_conv_relu, axis=0), cmap=cm.jet, interpolation='lanczos')
plt.axis('off')
plt.savefig(os.path.join(save_path, 'concat_conv_relu_sum.png'))
#plt.show()

for i in range(len(blobs)):
    blob = blobs[i]
    blob_name = blobs_name[i]
    channel = blob.shape[0]
    for c in range(channel):
#        plt.figure()
        feat = blob[c, :, :]
        plt.imshow(feat, cmap=cm.jet, interpolation='lanczos')
        plt.axis('off')
        plt.savefig(os.path.join(save_path, blob_name+'_'+str(c)+'.png'))
        
    