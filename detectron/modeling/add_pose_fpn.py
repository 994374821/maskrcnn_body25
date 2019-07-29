#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:52:47 2018

@author: gaomingda
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
import collections
import numpy as np
import logging

from detectron.modeling.generate_anchors import generate_anchors
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)

def add_pose_fpn_func(model, fpn_orig54, lateral_pose_numbers=1):
    """add pose to fpn
    return [fpn5, fpn4, fpn3_pose, fpn2_pose]
    """
    fpn_dim = cfg.FPN.DIM
    xavier_fill = ('XavierFill', {})
    
    fpn5, fpn4 = fpn_orig54
    blobs_out_fpn = [fpn5, fpn4, 'fpn3_pose', 'fpn2_pose'] # 0, 1, 2, 3
    
    fpn_level_info = fpn_level_info_ResNet50_conv5()
    lateral_input_blobs = fpn_level_info.blobs
#    # stop gradient
#    for i in range(len(lateral_input_blobs)):
#        model.StopGradient(s, s)
    fpn_dim_lateral = fpn_level_info.dims
    
    middle_blobs_lateral = ['fpn_inner_before_pose_{}'.format(s)
        for s in fpn_level_info.blobs]
    blobs_lateral = []
    # prepare pose predict blobs        
    hg_pred_NCHW_8 = model.net.NHWC2NCHW('pose_pred_8', 'hg_pred_NCHW_8')
    model.Relu(hg_pred_NCHW_8, hg_pred_NCHW_8)
    hg_pred_NCHW_4 = model.net.NHWC2NCHW('pose_pred_4', 'hg_pred_NCHW_4')
    model.Relu(hg_pred_NCHW_4, hg_pred_NCHW_4)
    hg_pred_NCHW = [hg_pred_NCHW_8, hg_pred_NCHW_4]
    
    for i in range(len(lateral_input_blobs)):
        if cfg.FPN.USE_GN:
            # use GroupNorm
            c = model.ConvGN(
                lateral_input_blobs[i],
                middle_blobs_lateral[i],  # note: this is a prefix
                dim_in=fpn_dim_lateral[i],
                dim_out=fpn_dim,
                group_gn=get_group_gn(fpn_dim),
                kernel=1,
                pad=0,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
            middle_blobs_lateral[i] = c  # rename it
        else:
            model.Conv(
                lateral_input_blobs[i],
                middle_blobs_lateral[i],
                dim_in=fpn_dim_lateral[i],
                dim_out=fpn_dim,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            ) 
        
        # add pose blob to lateral
        for j in range(lateral_pose_numbers):
            concat_name = 'fpn_inner_pose{}_concat_{}'.format(j,lateral_input_blobs[i])
            pose_conv_name = 'fpn_inner_pose{}_conv_{}'.format(j,lateral_input_blobs[i])
            if j==0:
                hg_pred_NCHW_concat, _ = model.net.Concat([middle_blobs_lateral[i], hg_pred_NCHW[i]], 
                                                          [concat_name, concat_name+'_info'],
                          axis=1)
            else:
                hg_pred_NCHW_concat, _ = model.net.Concat([blob_in, hg_pred_NCHW[i]], 
                                                          [concat_name, concat_name+'_info'],
                          axis=1)
            blob_in = model.Conv(
                 hg_pred_NCHW_concat, pose_conv_name, fpn_dim+16, fpn_dim, kernel=3, stride=1, pad=1,
                 weight_init=('XavierFill', {}),
                 bias_init=const_fill(0.0)
                 )
            model.Relu(blob_in, blob_in)
        blobs_lateral.append(blob_in)
    
    #####add top-down
    # Top-down 2x upsampling
    td = model.net.UpsampleNearest(fpn4, 'fpn4' + '_topdown_pose', scale=2)
    # Sum lateral and top-down
    model.net.Sum([blobs_lateral[0], td], 'fpn3_sum_pose')
    model.Conv(
             'fpn3_sum_pose', blobs_out_fpn[2], fpn_dim, fpn_dim, kernel=3, stride=1, pad=1,
             weight_init=('XavierFill', {}),
             bias_init=const_fill(0.0)
                 )
    td = model.net.UpsampleNearest(blobs_out_fpn[2], 'fpn3' + '_topdown_pose', scale=2)
    # Sum lateral and top-down
    model.net.Sum([blobs_lateral[1], td], 'fpn2_sum_pose')
    model.Conv(
             'fpn2_sum_pose', blobs_out_fpn[3], fpn_dim, fpn_dim, kernel=3, stride=1, pad=1,
             weight_init=('XavierFill', {}),
             bias_init=const_fill(0.0)
                 )
    
    
    return blobs_out_fpn
    
    


def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blob_in,
        'cls_score_pose',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score_pose', 'cls_prob_pose', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        blob_in,
        'bbox_pred_pose',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )


def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
#    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
#        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
#        scale=model.GetLossScale()
#    )
    
#     focal loss
    model.Softmax('cls_score_pose', 'cls_prob_pose', engine='CUDNN')
    model.net.Reshape(['cls_score_pose'], ['cls_score_pose_','cls_score_pose_shape'], 
                      shape=(1, -1, 1, 1) )
    model.net.Reshape(['labels_int32'], ['labels_int32_','labels_int32_shape'], 
                      shape=(1, -1, 1, 1)  )                  
#    normalizer = model.net.ConstantFill([], ['normalizer'], value=1., dtype=float32)
#    workspace.FeedBlob('gpu_0/normalizer', np.array([100], dtype=np.float32))
    loss_cls, cls_prob_ = model.net.SoftmaxFocalLoss(
        ['cls_score_pose_', 'labels_int32_', 'normalizer'], ['loss_cls_pose', 'cls_prob_focal_pose'],
        scale=model.GetLossScale(), gamma=2., num_classes=model.num_classes
    )
    
    loss_bbox = model.net.SmoothL1Loss(
        [
            'bbox_pred_pose', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox_pose',
        scale=model.GetLossScale()
    )

    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.AddLosses(['loss_cls_pose', 'loss_bbox_pose'])
    
    model.Accuracy(['cls_prob_pose', 'labels_int32'], 'accuracy_cls_pose')
    model.AddMetrics('accuracy_cls_pose')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat_pose_fpn',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    
    model.FC(roi_feat, 'fc6_pose', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6_pose', 'fc6_pose')
    model.FC('fc6_pose', 'fc7_pose', hidden_dim, hidden_dim)
    model.Relu('fc7_pose', 'fc7_pose')
    return 'fc7_pose', hidden_dim


FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)    
    
def fpn_level_info_ResNet50_conv5():
    return FpnLevelInfo(
        blobs=('res3_3_sum', 'res2_2_sum'),
        dims=(512, 256),
        spatial_scales=(1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )
    
def add_pose_fpn_rcnn_head(model, fpn_orig54, dim_conv, spatial_scale_conv):
    
    pose_fpn_blob = add_pose_fpn_func(model, fpn_orig54, lateral_pose_numbers=1)
    blob_in, dim = add_roi_2mlp_head(model, pose_fpn_blob, dim_conv, spatial_scale_conv)
    add_fast_rcnn_outputs(model, blob_in, dim)
    
    if model.train:
        loss_gradients = add_fast_rcnn_losses(model)
    else:
        loss_gradients = None
    return loss_gradients
    
    
    
    