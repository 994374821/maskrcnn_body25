# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils

from caffe2.python import workspace, core

import numpy as np


# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    if cfg.PRED_STD:
        model.FC(
            blob_in,
            'bbox_pred_std',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.0001),
            bias_init=const_fill(1.0)
        )
        model.net.Abs('bbox_pred_std', 'bbox_pred_std_abs')



def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
#    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
#        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
#        scale=model.GetLossScale()
#    )
    
#     focal loss
    model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    model.net.Reshape(['cls_score'], ['cls_score_','cls_score_shape'], 
                      shape=(1, -1, 1, 1) )
    model.net.Reshape(['labels_int32'], ['labels_int32_','labels_int32_shape'], 
                      shape=(1, -1, 1, 1)  )                  
    loss_cls, cls_prob_ = model.net.SoftmaxFocalLoss(
        ['cls_score_', 'labels_int32_', 'normalizer'], ['loss_cls', 'cls_prob_focal'],
        scale=model.GetLossScale(), gamma=2., num_classes=model.num_classes
    )
    if cfg.PRED_STD:
        #loss_bbox = model.net.KLLoss(
        #    [
        #        'bbox_pred','bbox_pred_std_abs', 'bbox_targets', 'bbox_inside_weights',
        #        
        #    ],
        #    'loss_bbox',
        #    scale=model.GetLossScale()
        #)
        #stop pred
        model.net.Copy('bbox_pred', 'bbox_pred0')
        #model.net.Copy('bbox_pred_std_abs', 'bbox_pred_std_abs0')
        #model.net.StopGradient('bbox_pred0', 'bbox_pred0')
        #model.net.StopGradient('bbox_pred_std_abs0', 'bbox_pred_std_abs0')
        #model.net.StopGradient('bbox_inside_weights', 'bbox_inside_weights')
        #model.net.StopGradient('bbox_targets', 'bbox_targets')
        ################# bbox_std grad, stop pred
        #log(std)
        model.net.ConstantFill([], 'sigma', value=0.0001, shape=(1,))
        model.net.Add(['bbox_pred_std_abs', 'sigma'], 'bbox_pred_std_abs_')
        model.net.Log('bbox_pred_std_abs_', 'bbox_pred_std_abs_log_')
        model.net.Scale('bbox_pred_std_abs_log_', 'bbox_pred_std_abs_log', scale=-0.5*model.GetLossScale())
        model.net.StopGradient('bbox_outside_weights', 'bbox_outside_weights')
        model.net.Mul(['bbox_pred_std_abs_log', 'bbox_outside_weights'], 'bbox_pred_std_abs_logw')
        model.net.ReduceMean('bbox_pred_std_abs_logw', 'bbox_pred_std_abs_logwr', axes=[0])
        bbox_pred_std_abs_logw_loss = model.net.SumElements(
                'bbox_pred_std_abs_logwr', 'bbox_pred_std_abs_logw_loss')
        #pred0 - y
        model.net.Sub(['bbox_pred0', 'bbox_targets'], 'bbox_in')
        #val = in*(pred0 - u)
        model.net.Mul(['bbox_in', 'bbox_inside_weights'], 'bbox_inw')
        #absval
        model.net.Abs('bbox_inw', 'bbox_l1abs')
        #l12 mask
        model.net.ConstantFill([], 'one', value=1., shape=(1,))
        model.net.GE(['bbox_l1abs', 'one'], 'wl1', broadcast=1)
        model.net.LT(['bbox_l1abs', 'one'], 'wl2', broadcast=1)
        model.net.Cast('wl1', 'wl1f', to=core.DataType.FLOAT) 
        model.net.Cast('wl2', 'wl2f', to=core.DataType.FLOAT)
        #model.net.StopGradient('wl1f', 'wl1f')
        #model.net.StopGradient('wl2f', 'wl2f')
        # val^2
        model.net.Mul(['bbox_inw', 'bbox_inw'], 'bbox_sq')
        # 0.5 val^2
        model.net.Mul(['bbox_sq', 'wl2f'], 'bbox_l2_')
        model.net.Scale('bbox_l2_', 'bbox_l2', scale=0.5)
        # absval - 0.5
        model.net.ConstantFill([], 'half', value=.5, shape=(1,))
        model.net.Sub(['bbox_l1abs', 'half'], 'bbox_l1abs_', broadcast=1)
        model.net.Mul(['bbox_l1abs_', 'wl1f'], 'bbox_l1')
        # sml1 = w * l1 + w*l2
        model.net.Add(['bbox_l1', 'bbox_l2'], 'bbox_inws')
        #alpha * sml1
        model.net.StopGradient('bbox_inws', 'bbox_inws')
        model.net.Mul(['bbox_pred_std_abs', 'bbox_inws'], 'bbox_inws_out')
        model.net.Scale('bbox_inws_out', 'bbox_inws_out', scale=model.GetLossScale())
        model.net.ReduceMean('bbox_inws_out', 'bbox_inws_outr', axes=[0])
        bbox_pred_std_abs_mulw_loss = model.net.SumElements(
                ['bbox_inws_outr'], 'bbox_pred_std_abs_mulw_loss')

        #bbox_pred grad, stop std
        loss_bbox = model.net.SmoothL1Loss(
            [
                'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
                'bbox_pred_std_abs'
            ],
            'loss_bbox',
            scale=model.GetLossScale()
        )
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox, 
            bbox_pred_std_abs_mulw_loss, 
            bbox_pred_std_abs_logw_loss
            ])
        model.AddLosses(['loss_cls', 'loss_bbox', 
            'bbox_pred_std_abs_mulw_loss', 'bbox_pred_std_abs_logw_loss'
            ])
    else:
        loss_bbox = model.net.SmoothL1Loss(
            [
                'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
                'bbox_outside_weights'
            ],
            'loss_bbox',
            scale=model.GetLossScale()
        )

        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
        model.AddLosses(['loss_cls', 'loss_bbox'])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddMetrics('accuracy_cls')
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
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    
# ==============================pose===============================================
#    model.net.NHWC2NCHW('pose_pred', 'hg_pred_NCHW')
#    roi_heatmap_feat = model.RoIFeatureTransform(
#             ['hg_pred_NCHW','hg_pred_NCHW','hg_pred_NCHW','hg_pred_NCHW'],
#             'roi_heatmap_feat',
#             blob_rois='rois_hg',
#             method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
#             resolution=roi_size,
#             sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
#             spatial_scale=[1./8, 1./8, 1./8, 1./8]
#         )
#    model.net.Concat([roi_feat, roi_heatmap_feat], ['roi_heatmap_concat_feat', 'roi_heatmap_info'],
#                      axis=1)
#    roi_feat_pose = model.Conv(
#             'roi_heatmap_concat_feat', 'roi_feat_pose', (256+16), 256, kernel=3, stride=1, pad=1,
#             weight_init=('XavierFill', {}),
#             bias_init=const_fill(0.0)
#             )
# #    roi_feat_pose_bn = model.AffineChannel(roi_feat_pose, 'roi_feat_pose_bn', dim=256, 
# #                                            share_with=False, inplace=True)
# #    roi_feat_pose_bn = model.Relu(roi_feat_pose_bn, roi_feat_pose_bn)
#    model.FC(roi_feat_pose, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
# =============================================================================
# ==============================mask restop===============================================
#    roi_heatmap_feat = model.RoIFeatureTransform(
#             ['human_conv1','human_conv1','human_conv1','human_conv1'],
#             'roi_heatmap_feat',
#             blob_rois='rois',
#             method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
#             resolution=roi_size,
#             sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
#             spatial_scale=[1./4, 1./4, 1./4, 1./4]
#         )
#    model.net.Concat([roi_feat, roi_heatmap_feat], ['roi_heatmap_concat_feat', 'roi_heatmap_info'],
#                      axis=1)
#    roi_feat_pose = model.Conv(
#             'roi_heatmap_concat_feat', 'roi_feat_pose', (256+256), 256, kernel=3, stride=1, pad=1,
#             weight_init=('XavierFill', {}),
#             bias_init=const_fill(0.0)
#             )
# #    roi_feat_pose_bn = model.AffineChannel(roi_feat_pose, 'roi_feat_pose_bn', dim=256, 
# #                                            share_with=False, inplace=True)
# #    roi_feat_pose_bn = model.Relu(roi_feat_pose_bn, roi_feat_pose_bn)
#    model.FC(roi_feat_pose, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
# =============================================================================

    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim
