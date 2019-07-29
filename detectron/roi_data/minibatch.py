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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Construct minibatches for Detectron networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import numpy as np
import os
import copy

from detectron.core.config import cfg
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.retinanet as retinanet_roi_data
import detectron.roi_data.rpn as rpn_roi_data
import detectron.utils.blob as blob_utils

logger = logging.getLogger(__name__)

#import sys
#sys.path.append('/home/gaomingda/softer_nms_LIP_JPPNet/detectron/LIP_JPPNet')
#from evaluate_pose_JPPNet import draw_resized_pose

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    blob_names += ['normalizer'] # focal loss at fast_rcnn_heads
#    blob_names += ['normalizer_fcn'] # focal loss at mask_res_top
#    blob_names += ['pose_pred']
    blob_names += ['pose_pred_4']
    blob_names += ['pose_pred_8']
    blob_names += ['pose_pred_16']
    blob_names += ['pose_pred_32']
    
    blob_names += ['pose_line_8']
    blob_names += ['pose_line_16']
    
    # seg_gt_label, add segementation on top of fpn2-5
    blob_names += ['seg_gt_label']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += rpn_roi_data.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        blob_names += retinanet_roi_data.get_retinanet_blob_names(
            is_training=is_training
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += fast_rcnn_roi_data.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


#def get_minibatch(roidb, pose_pred_model):
def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
#    im_blob, im_scales = _get_image_blob(roidb)
    im_blob, im_scales, pose_pred, pose_line, blobs['seg_gt_label'] = _get_image_pose_blob(roidb) # pose_pred the same shape with im_blob
    blobs['data'] = im_blob
    blobs['normalizer'] = np.array([100], dtype=np.float32)
    if 'LIP' in cfg.TRAIN.DATASETS[0]:
        blobs['pose_pred_4'], blobs['pose_pred_8'], blobs['pose_pred_16'], blobs['pose_pred_32'] = _resize_pose_blob(pose_pred, channel=26)
    else:
        blobs['pose_pred_4'], blobs['pose_pred_8'], blobs['pose_pred_16'], blobs['pose_pred_32'] = _resize_pose_blob(pose_pred, channel=26)
#    blobs['pose_pred_8'], blobs['pose_pred_16'] = _resize_pose_blob_to13(pose_pred) # pose 16 to 13 channel
#    blobs['pose_sum_8'], blobs['pose_sum_16'] = pose_sum_to_onehotmap(blobs['pose_pred_8'], blobs['pose_pred_16'])
    blobs['pose_line_8'], blobs['pose_line_16'] = _resize_poseline_blob(pose_line)
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = rpn_roi_data.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        im_width, im_height = im_blob.shape[3], im_blob.shape[2]
        # im_width, im_height corresponds to the network input: padded image
        # (if needed) width and height. We pass it as input and slice the data
        # accordingly so that we don't need to use SampleAsOp
        valid = retinanet_roi_data.add_retinanet_blobs(
            blobs, im_scales, roidb, im_width, im_height
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = fast_rcnn_roi_data.add_fast_rcnn_blobs(blobs, im_scales, roidb)
#    blobs['pose_pred'] = pose_pred_model.pred_pose_batch(roidb)
#    pose_pred_model.draw_batch(blobs['pose_pred'], roidb)
#    blobs['pose_pred'] = _get_pose_pred(roidb)
    
#    logger.info(blobs['pose_pred'].shape)
    return blobs, valid


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

def _get_pose_pred(roidb, channel=16):
    """get LIP_JPP pose prediction from .bin file
    """
    num_images = len(roidb)
    if 'LIP' in cfg.TRAIN.DATASETS[0]:
#        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/LIP_JPP_pred_pose/train'
        pred_pose_data = '/home/gaomingda/datasets/lip_body25/train_images'
        pose_line_data = '/home/gaomingda/Downloads/gaomingda/dataset/LIP_JPP_pose_edge/train'
    if 'ATR' in cfg.TRAIN.DATASETS[0]:
#        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/ATR_JPP_pred_pose'
#        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/ATR_JPP_crop_pred_pose'
        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/ATR_openpose'
    if 'LIP' in cfg.TRAIN.DATASETS[0]:
        pose_blob = np.zeros((num_images, 48, 48, channel), dtype=np.float32)
    else: # ATR
        pose_blob = np.zeros((num_images, 48, 48, 26), dtype=np.float32)
    pose_line_blob = np.zeros((num_images, 48, 48), dtype=np.float32)
    for i in range(num_images):
        entry = roidb[i]
        if 'ATR' in cfg.TRAIN.DATASETS[0]:
            if entry['flipped']:
                pred_pose_path = os.path.join(pred_pose_data, 'heatmap_flip', entry['id']+'.bin')
            else:
                pred_pose_path = os.path.join(pred_pose_data, 'heatmap', entry['id']+'.bin')
            pred_ = np.fromfile(pred_pose_path, dtype=np.float32)
            pred_ = pred_.reshape(48, 48, 26)
            pose_blob[i] = pred_
        else: # LIP
            if entry['flipped']:
                pred_pose_path = os.path.join(pred_pose_data, 'heatmap_flip', entry['id']+'.bin')
            else:
                pred_pose_path = os.path.join(pred_pose_data, 'heatmap', entry['id']+'.bin')
            pred_ = np.fromfile(pred_pose_path, dtype=np.float32)
            pred_ = pred_.reshape(48, 48, channel)
            # pose line
            #pose_line = np.fromfile(os.path.join(pose_line_data, entry['id']+'.bin'), dtype=np.float32)
            #pose_line = pose_line.reshape(48, 48)
            pose_line = np.zeros((48, 48), dtype=np.float32)
#            if entry['flipped']:
#                pred_ = flip_pose(pred_)
#                # pose line
#                pose_line = pose_line[:, ::-1]
            pose_blob[i] = pred_
            # pose line 
            pose_line_blob[i] = pose_line
    # select 0-15 channel
    #print("train body25, select poses 0-16 channel")
    pose_blob = pose_blob[:, :, :, 0:16]
    return pose_blob , pose_line_blob

def flip_pose(pose):
    """input: pose, is array of size(none, none, 16)
    """
    flip_pose = np.zeros(pose.shape, dtype=np.float32)
    flip_pose[:, :, 0] = pose[:, :, 5]
    flip_pose[:, :, 1] = pose[:, :, 4]
    flip_pose[:, :, 2] = pose[:, :, 3]
    flip_pose[:, :, 3] = pose[:, :, 2]
    flip_pose[:, :, 4] = pose[:, :, 1]
    flip_pose[:, :, 5] = pose[:, :, 0]
    flip_pose[:, :, 10] = pose[:, :, 15]
    flip_pose[:, :, 11] = pose[:, :, 14]
    flip_pose[:, :, 12] = pose[:, :, 13]
    flip_pose[:, :, 13] = pose[:, :, 12]
    flip_pose[:, :, 14] = pose[:, :, 11]
    flip_pose[:, :, 15] = pose[:, :, 10]
    flip_pose[:, :, 6] = pose[:, :, 6]
    flip_pose[:, :, 7] = pose[:, :, 7]
    flip_pose[:, :, 8] = pose[:, :, 8]
    flip_pose[:, :, 9] = pose[:, :, 9]
    return flip_pose[:, ::-1, :]


def _get_image_pose_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    # pose_line: (num_images, 48, 48)
    poses, pose_line = _get_pose_pred(roidb,  channel=26)
#    show_pose(roidb, poses, im_scales)
    # seg_gt label
    seg_gt_list = _prep_seg_gt_for_blob(roidb)

    im_blob, pose_blob, pose_line_blob, seg_gt_blob = blob_utils.im_list_to_blob_andPose(processed_ims, poses, pose_line, seg_gt_list)
#    show_pose(roidb, pose_blob, im_scales)
#    blob = blob_utils.im_list_to_blob(processed_ims)

    return im_blob, im_scales, pose_blob, pose_line_blob, seg_gt_blob

def show_pose(roidb, pose_blob, im_scales):
    num_images = len(roidb)
    for i in range(num_images):
        pose_blob_i = pose_blob[i]
        pred_poses = []
        for j in range(16):
            channel_ = pose_blob_i[:, :, j]
            r_, c_ = np.unravel_index(channel_.argmax(), channel_.shape)
#            if channel_[r_, c_]>0.3:
#                pred_poses.append([r_, c_])
#            else:
#                pred_poses.append([-1, -1])
            pred_poses.append([r_, c_])
        draw_resized_pose(roidb[i]['image'], pred_poses, im_scales[i], roidb[i]['flipped'])

def _resize_pose_blob(pose_pred, channel=16):
    n, h, w, channel = pose_pred.shape
    pose_shrink4 = np.zeros((n, int(h/4.), int(w/4.), channel), dtype=np.float32)
    pose_shrink8 = np.zeros((n, int(h/8.), int(w/8.), channel), dtype=np.float32)
    pose_shrink16 = np.zeros((n, int(h/16.), int(w/16.), channel), dtype=np.float32)
    pose_shrink32 = np.zeros((n, int(h/32.), int(w/32.), channel), dtype=np.float32)
    for i in range(n):
        pose_shrink4[i] = cv2.resize(pose_pred[i, :, :, :], None, None, 1./4., 1./4.,
                    interpolation=cv2.INTER_LINEAR)
        pose_shrink8[i] = cv2.resize(pose_pred[i, :, :, :], None, None, 1./8., 1./8.,
                    interpolation=cv2.INTER_LINEAR)
        pose_shrink16[i] = cv2.resize(pose_pred[i], None, None, 1./16., 1./16.,
                    interpolation=cv2.INTER_LINEAR)
        pose_shrink32[i] = cv2.resize(pose_pred[i], None, None, 1./32., 1./32.,
                    interpolation=cv2.INTER_LINEAR)
    return pose_shrink4, pose_shrink8, pose_shrink16, pose_shrink32

def _resize_poseline_blob(pose_pred):
    n, h, w = pose_pred.shape
#    pose_shrink4 = np.zeros((n, int(h/4.), int(w/4.)), dtype=np.float32)
    pose_shrink8 = np.zeros((n, int(h/8.), int(w/8.)), dtype=np.float32)
    pose_shrink16 = np.zeros((n, int(h/16.), int(w/16.)), dtype=np.float32)
    for i in range(n):
#        pose_shrink4[i] = cv2.resize(pose_pred[i, :, :], None, None, 1./4., 1./4.,
#                    interpolation=cv2.INTER_NEAREST)
        pose_shrink8[i] = cv2.resize(pose_pred[i, :, :], None, None, 1./8., 1./8.,
                    interpolation=cv2.INTER_NEAREST)
        pose_shrink16[i] = cv2.resize(pose_pred[i], None, None, 1./16., 1./16.,
                    interpolation=cv2.INTER_NEAREST)
    return  pose_shrink8, pose_shrink16

def _resize_pose_blob_to13(pose_pred):
    """first combine 16 channel pose pred to 13 channel(6,B_Pelvis ,B_Spine ,B_Neck ,B_Head)
    then shrink 1./8, 1.16 the 13 channel pose blob 
    """
    n, h, w, _ = pose_pred.shape
    pose_13 = np.zeros((n, h, w, 13), dtype=np.float32)
    pose_13[:, :, :, 0:6] = pose_pred[:, :, :, 0:6]
    pose_13[:, :, :, 7] = pose_pred[:, :, :, 10]
    pose_13[:, :, :, 6] = pose_pred[:, :, :, 6]+pose_pred[:, :, :, 7]+pose_pred[:, :, :, 8]+pose_pred[:, :, :, 9]
    return _resize_pose_blob(pose_13, channel=13)

def pose_sum_to_onehotmap(pose_blob_8, pose_blob_16):
    """pose_blob: shape (num_imgs, h, w, 16)
        pose_blob_8: same shape with res3
        pose_blob_16: same shape with res4
    """
    n, h, w, c = pose_blob_8.shape
    _, h_16, w_16, _ = pose_blob_16.shape
    
    pose_sum_8 = np.sum(pose_blob_8, axis=3)
    pose_sum_16 = np.sum(pose_blob_16, axis=3)
    
    one_hot_blob_res3 = np.zeros((n, 512, h, w), dtype=np.float32)
    one_hot_blob_res4 = np.zeros((n, 1024, h_16, w_16), dtype=np.float32)
    for i in range(n):
        one_hot_blob_res3[i, :, :, :] = pose_sum_8[i]
        one_hot_blob_res4[i, :, :, :] = pose_sum_16[i]
    return one_hot_blob_res3, one_hot_blob_res4

def _prep_seg_gt_for_blob(roidb):
    """load seg gt label
       return: 2D array
       return: a list of seg_gt array(H, W)
    """
    seg_gt_list = []
    for entry in roidb:
        seg_gt = cv2.imread(entry['ins_seg'], 0)
        if entry['flipped']:
            seg_gt = seg_gt[:, ::-1]
            label_ = copy.deepcopy(seg_gt)
            dataset_name = cfg.TRAIN.DATASETS[0]
            if 'LIP' in dataset_name:
                orig2flipped = {14:15, 15:14, 16:17, 17:16, 18:19, 19:18}
            if 'ATR' in dataset_name:
                orig2flipped = {
                            9: 10, 10: 9, 12: 13, 13: 12, 14: 15, 15: 14}
    
            for i in orig2flipped.keys():
                ind_i = np.where(label_==i)
                if len(ind_i[0])==0:
                    continue
                seg_gt[ind_i] = int(orig2flipped[i])
                
    #    seg_gt = cv2.resize(seg_gt, None, None, fx=im_scale, fy=im_scale,
    #                        interpolation=cv2.INTER_NEAREST)
        seg_gt = np.array(seg_gt, dtype=np.int32)
        seg_gt_list.append(seg_gt)

    return seg_gt_list     