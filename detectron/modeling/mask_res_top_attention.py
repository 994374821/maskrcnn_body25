#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:48:18 2018

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

def conv_bn_relu(model, blob_in, dim_in, dim_out, name, k=1, p=0, s=1):
    p = model.Conv(blob_in, name, dim_in, dim_out, kernel=k, pad=p, stride=s, no_bias=1, weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}))
    p = model.AffineChannel(p, name+'_bn', dim=dim_out, inplace=True)
    p = model.Relu(p, p)
    return p

def PAM(model, A, A_channel):
    """position attention model
    """
    out_channel = A_channel//2
    # down sample
    model.net.MaxPool(A, 'A_pool1', kernel=3, pad=1, stride=2)
    model.net.Shape('A_pool1', 'A_shape')
    B = conv_bn_relu(model, 'A_pool1', A_channel, out_channel, 'B')
    C = conv_bn_relu(model, 'A_pool1', A_channel, out_channel, 'C')
    D = conv_bn_relu(model, 'A_pool1', A_channel, A_channel, 'D')
    
    if model.train:
        N = cfg.TRAIN.IMS_PER_BATCH
    else:
        N = 1
    model.net.Reshape(B, ['B_reshape','B_shape'], shape=(N, out_channel, -1))
    model.net.Reshape(C, ['C_reshape', ' C_shape'], shape=(N, out_channel, -1))
    model.net.Reshape(D, ['D_reshape', 'D_shape'], shape=(N, A_channel, -1))
    
    model.net.BatchMatMul(['B_reshape', 'C_reshape'], ['S'], trans_a=1)
    model.net.Softmax('S', 'S_probs', axis=2)
    
    model.net.BatchMatMul(['D_reshape', 'S'], ['D_S'])
    model.net.Reshape(['D_S', 'A_shape'], ['D_S_reshape', 'D_S_shape'])
    # upsample
    model.BilinearInterpolation(
                'D_S_reshape', 'D_S_reshape_up', A_channel, A_channel, 2
            )
    model.net.Scale('D_S_reshape_up', 'D_S_reshape_up', scale=0.0000001)
    
    return model.net.Add(['D_S_reshape_up', A], 'PA')

def PSPBlob(model, blob_in, dim_in):    
    model.net.AveragePool2D(blob_in, 'psp_pool1', kernel=30, stride=30, pad=30)
    model.net.AveragePool2D(blob_in, 'psp_pool2', kernel=20, stride=20, pad=20)
    model.net.AveragePool2D(blob_in, 'psp_pool3', kernel=10, stride=10, pad=10)
    model.net.AveragePool2D(blob_in, 'psp_pool4', kernel=4, stride=4, pad=0)
    
    pool1_conv = conv_bn_relu(model, 'psp_pool1', dim_in, dim_in//2, 'psp_pool1_conv')
    pool2_conv = conv_bn_relu(model, 'psp_pool2', dim_in, dim_in//2, 'psp_pool2_conv')
    pool3_conv = conv_bn_relu(model, 'psp_pool3', dim_in, dim_in//2, 'psp_pool3_conv')
    pool4_conv = conv_bn_relu(model, 'psp_pool4', dim_in, dim_in//2, 'psp_pool4_conv')
    
#    pool1_conv_interp = model.net.ResizeLike([pool1_conv, tempt], 'psp_pool1_conv_interp')
#    pool2_conv_interp = model.net.ResizeLike([pool2_conv, tempt], 'psp_pool2_conv_interp')
#    pool3_conv_interp = model.net.ResizeLike([pool3_conv, tempt], 'psp_pool3_conv_interp')
#    pool4_conv_interp = model.net.ResizeLike([pool4_conv, tempt], 'psp_pool4_conv_interp')
    pool1_conv_interp = model.net.ResizeNearestLike([pool1_conv, blob_in], 'psp_pool1_conv_interp')
    pool2_conv_interp = model.net.ResizeNearestLike([pool2_conv, blob_in], 'psp_pool2_conv_interp')
    pool3_conv_interp = model.net.ResizeNearestLike([pool3_conv, blob_in], 'psp_pool3_conv_interp')
    pool4_conv_interp = model.net.ResizeNearestLike([pool4_conv, blob_in], 'psp_pool4_conv_interp')
    
    model.net.Concat([blob_in, pool1_conv_interp, pool2_conv_interp, pool3_conv_interp, pool4_conv_interp], 
                     ['psp_pool_concat', 'pool_concat_info'], axis=1)
    
    p = conv_bn_relu(model, 'psp_pool_concat', dim_in*4//2+dim_in, dim_in, 'psp_concat_conv', k=3, p=1, s=1)
    
    return p
    
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
    # pose
#    p, _ = model.net.Concat([p, 'hg_pred_NCHW_4'], ['restop_pose_concat', 'restop_pose_info'], axis=1)
#    dim_in += 26
    
#    p = PAM(model, p, 256)
#    p = PSPBlob(model, p, 256)
    
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