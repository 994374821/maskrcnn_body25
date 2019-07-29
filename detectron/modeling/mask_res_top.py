#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:08:53 2018

@author: gaomingda
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.utils.blob as blob_utils

def mask_res_top_head(model, blob_in, dim_in):
    """blob_in: [fpn5, fpn4, fpn3, fpn2]
    dim_in: 256(default)
    """
    blob_in_up = ['fpn{}_up'.format(5-i) for i in range(3)]
    up_scale = [8, 4, 2]
    for i in range(3):
        model.net.UpsampleNearest(blob_in[i], blob_in_up[i], scale=up_scale[i])
    blob_in_up.append(blob_in[-1])
    p, _ = model.net.Concat(blob_in_up, ['fpn_concat', 'fpn_info'], axis=1)
    p = model.Conv(p, 'human_conv1', 256*4, 256, kernel=3, pad=1, stride=1, no_bias=1, weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}))
    p = model.AffineChannel(p, 'human_conv1_bn', dim=dim_in, inplace=True)
    p = model.Relu(p, p)
    
    human_fc = model.Conv(
            p,
            'human_fc',
            dim_in,
            model.num_classes,
            kernel=1,
            pad=0,
            stride=1,
#            dilation=6,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    '''
    model.Conv(
            'human_fc',
            'human_fc_fc1',
            dim_in,
            model.num_classes,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    # 12
    model.Conv(
            p,
            'res_fc1_human',
            dim_in,
            20,
            kernel=3,
            pad=1 * 12,
            stride=1,
            dilation=12,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
    model.Conv(
            'res_fc1_human',
            'res_fc1_human_fc1',
            dim_in,
            model.num_classes,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    # 18
    model.Conv(
            p,
            'res_fc2_human',
            dim_in,
            20,
            kernel=3,
            pad=1 * 18,
            stride=1,
            dilation=18,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
    model.Conv(
            'res_fc2_human',
            'res_fc2_human_fc1',
            dim_in,
            model.num_classes,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    # 24
    model.Conv(
            p,
            'res_fc3_human',
            dim_in,
            20,
            kernel=3,
            pad=1 * 24,
            stride=1,
            dilation=24,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
    model.Conv(
            'res_fc3_human',
            'res_fc3_human_fc1',
            dim_in,
            model.num_classes,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    
    model.Add(['human_fc_fc1', 'res_fc1_human_fc1'], 'res_fc_human1')
    model.Add(['res_fc2_human_fc1', 'res_fc3_human_fc1'], 'res_fc_human2')
    human_fc = model.Add(['res_fc_human2', 'res_fc_human1'], 'res_fc_human3')
    '''
    
    if not model.train:
        model.net.NCHW2NHWC(human_fc, 'seg_score_NHWC')
        model.net.Softmax('seg_score_NHWC', 'probs_human_NHWC', axis=3)
        model.net.NHWC2NCHW('probs_human_NHWC', 'probs_human_NCHW')
    
    return human_fc

def add_mask_res_loss(model, blob_mask_res):
    
    model.net.NCHW2NHWC(blob_mask_res, 'seg_score_NHWC')
    model.Reshape('seg_score_NHWC', ['seg_score_reshape', 'seg_score_old_shape'], shape=[-1, model.num_classes])
    model.Reshape('seg_gt_label', ['seg_gt_label_reshape', 'seg_gt_label_shape'], shape=[-1,])
    
    probs_human, loss_human = model.net.SoftmaxWithLoss(
                            ['seg_score_reshape', 'seg_gt_label_reshape'], ['probs_human', 'loss_human'],
                            scale=1. / cfg.NUM_GPUS)
    
    '''
    # focal loss
    loss_human, probs_human = model.net.SoftmaxFocalLoss(
        [blob_mask_res, 'seg_gt_label', 'normalizer'], ['loss_human', 'probs_human'],
        scale=model.GetLossScale(), gamma=2., num_classes=model.num_classes
    )
    '''
    
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_human])
    model.AddLosses('loss_human')
    
    return loss_gradients

def add_mask_res_branch(model, blob_in, dim_in):
    loss_gradient = None
    human_fc = mask_res_top_head(model, blob_in, dim_in)
    if model.train:
        loss_gradient = add_mask_res_loss(model, human_fc)    
    return loss_gradient