#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 13:16:39 2018

@author: gaomingda
"""
from caffe2.python import core, workspace

from detectron.core.config import get_output_dir
from detectron.core.config import cfg
from detectron.core.test import _get_blobs

import numpy as np
import cv2
import os

def res_top_result(model, entry, save_png=True):
    im_info = workspace.FetchBlob(core.ScopedName('im_info'))
    blob_h, blob_w, scale = im_info[0] 
    
    img = cv2.imread(entry['image'])
    img_h, img_w, _ = img.shape
    
    probs_human = workspace.FetchBlob(core.ScopedName('probs_human_NCHW'))[0, :, :, :]
    probs_crop = np.zeros((probs_human.shape[0], int(img_h*scale), int(img_w*scale)), dtype=np.float32)
    for i in range(probs_crop.shape[0]):
        probs_up4 = cv2.resize(probs_human[i], None, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        probs_crop[i, :, :] = probs_up4[0:int(img_h*scale), 0:int(img_w*scale)]
    
    probs_resize = np.zeros((probs_crop.shape[0], int(img_h), int(img_w)), dtype=np.float32)
    for i in range(probs_crop.shape[0]):
        probs_resize[i, :, :] = cv2.resize(probs_crop[i], (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    
    dataset_name = cfg.TEST.DATASETS[0]
    output_dir = os.path.join(get_output_dir(dataset_name, training=False), 'vis_res_top')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
#    print(output_dir)
    if save_png:
      cv2.imwrite(os.path.join(output_dir, entry['id']+'.png'), probs_resize.argmax(0))
    
    return probs_resize
    
def feedBlob_run(model, im, entry):
    inputs, im_scale = _get_blobs(im, None, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, pose_model=None, entry=entry)
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.net.Proto().name)           