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

"""Caffe2 blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import cv2
import numpy as np
import os

from caffe2.proto import caffe2_pb2

from detectron.core.config import cfg


def get_image_blob(im, target_scale, target_max_size):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    processed_im, im_scale = prep_im_for_blob(
        im, cfg.PIXEL_MEANS, target_scale, target_max_size
    )
    blob = im_list_to_blob(processed_im)
    # NOTE: this height and width may be larger than actual scaled input image
    # due to the FPN.COARSEST_STRIDE related padding in im_list_to_blob. We are
    # maintaining this behavior for now to make existing results exactly
    # reproducible (in practice using the true input image height and width
    # yields nearly the same results, but they are sometimes slightly different
    # because predictions near the edge of the image will be pruned more
    # aggressively).
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, im_scale))[np.newaxis, :]
    return blob, im_scale, im_info.astype(np.float32)

# =====================when test stage, pose input, image blob========================================================
def get_image_pose_blob(im, target_scale, target_max_size, entry):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
        pose_blob: same shape with blob(ims), channel: 16
    """
    processed_im, im_scale = prep_im_for_blob(
        im, cfg.PIXEL_MEANS, target_scale, target_max_size
    )
    ###### load pose data from .bin file
    if 'LIP_val' in cfg.TEST.DATASETS[0]:
#        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/LIP_JPP_pred_pose/val'
#        pose_line_data = '/home/gaomingda/Downloads/gaomingda/dataset/LIP_JPP_pose_edge/val'
        pred_pose_data = '/home/gaomingda/datasets/lip_body25/val_images'
    if 'LIP_test' in cfg.TEST.DATASETS[0]:
        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/lip_body25/testing_images'
    if 'ATR' in cfg.TEST.DATASETS[0]:
#        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/ATR_JPP_pred_pose'
#        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/ATR_JPP_crop_pred_pose'
        pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/ATR_openpose'
#    pred_pose_data = '/home/gaomingda/Downloads/gaomingda/dataset/LIP_JPP_pred_pose/val'
    pred_pose_path = os.path.join(pred_pose_data, 'heatmap', entry['id']+'.bin')
    pred_ = np.fromfile(pred_pose_path, dtype=np.float32)
    if 'LIP' in cfg.TEST.DATASETS[0]:
        pred_ = pred_.reshape(48, 48, 26)
    else:
        pred_ = pred_.reshape(48, 48, 26)
    choose_flag = -1
    if choose_flag==0:
      # select 0-16 channel
      print("select 0-16 channel")
      pred_ = pred_[:, :, 0:16]
    if choose_flag==1:
      # arm 2,3,4,5,6,7
      print("set arm ids pose to 0")
      arm_ids = [2,3,4,5,6,7]
      for arm_id in arm_ids:
        pred_[:, :, arm_id] = 0
    if choose_flag==2:
      print("set leg ids pose to 0")
      leg_ids = [9,10,12,13]
      for arm_id in leg_ids:
        pred_[:, :, arm_id] = 0
    if choose_flag==3:
      print("set foot ids pose to 0")
      foot_ids = [11,22,23,24,14,19,20,21]
      for arm_id in foot_ids:
        pred_[:, :, arm_id] = 0
     # pose line
#    pred_pose_path = os.path.join(pose_line_data, entry['id']+'.bin')
#    pose_line = np.fromfile(pred_pose_path, dtype=np.float32)
#    pose_line = pose_line.reshape(48, 48)
    pose_line = np.zeros((48, 48), dtype=np.float32)
    
    blob, pose_blob, pose_line_blob = im_list_to_blob_andPose(processed_im, np.expand_dims(pred_, axis=0), 
                                              np.expand_dims(pose_line, axis=0))
   
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, im_scale))[np.newaxis, :]
    return blob, im_scale, im_info.astype(np.float32), pose_blob, pose_line_blob 
 
# =============================================================================
def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)

    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32
    )
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def im_list_to_blob_andPose(ims, poses, pose_line, seg_gt_list=None):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)

    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32
    )
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    if 'LIP' in cfg.TRAIN.DATASETS[0]:
        pose_blob = pose_to_blob(ims, poses, max_shape, channel=26)
    else:
        pose_blob = pose_to_blob(ims, poses, max_shape, channel=26)
    pose_line_blob = poseline_to_blob(ims, pose_line, max_shape)
    if seg_gt_list is not None:
        seg_gt_blob = seg_gt_to_blob(ims, seg_gt_list, max_shape)
        return blob, pose_blob, pose_line_blob, seg_gt_blob
    return blob, pose_blob, pose_line_blob

def pose_to_blob(ims, poses, max_shape, channel=16):
    """ims: list
       poses: array shape: (num_imgs, 48, 48, 16)
       max_shape: reurn blob's max (h, w)
    """
    num_images = len(ims)
    channel = poses[0].shape[-1]
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], channel), dtype=np.float32
    )
    for i in range(num_images):
        h_i, w_i, _ = ims[i].shape
        for j in range(channel):
            pose_i_j = poses[i, :, :, j]
            blob[i, 0:h_i, 0:w_i, j] = cv2.resize(pose_i_j, (w_i, h_i), interpolation=cv2.INTER_LINEAR)
    return blob

def poseline_to_blob(ims, poses_line, max_shape):
    """ims: list
       poses_line: array shape: (num_imgs, 48, 48)
       max_shape: reurn blob's max (h, w)
    """
    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1]), dtype=np.float32
    )
    for i in range(num_images):
        h_i, w_i, _ = ims[i].shape
        pose_i_j = poses_line[i, :, :]
        blob[i, 0:h_i, 0:w_i] = cv2.resize(pose_i_j, (w_i, h_i), interpolation=cv2.INTER_NEAREST)
    return blob

def seg_gt_to_blob(ims, seg_gt_list, max_shape):
    """ims: list
       seg_gt_list: list of array shape(48, 48)
       max_shape: reurn blob's max (h, w)
    """
    num_images = len(ims)
    blob = np.zeros(
        (num_images, int(max_shape[0]/4.0), int(max_shape[1]/4.)), dtype=np.int32
    )
    for i in range(num_images):
        h_i, w_i, _ = ims[i].shape
        pose_i_j = seg_gt_list[i]
        seg_gt_i = np.zeros((max_shape[0], max_shape[1]), dtype=np.int32)
        seg_gt_i[0:h_i, 0:w_i] = cv2.resize(pose_i_j, (w_i, h_i), interpolation=cv2.INTER_NEAREST)
    
        blob[i, :, : ] = cv2.resize(seg_gt_i, None, None, fx=1./4, fy=1./4, interpolation=cv2.INTER_NEAREST)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    return im, im_scale


def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


def py_op_copy_blob(blob_in, blob_out):
    """Copy a numpy ndarray given as blob_in into the Caffe2 CPUTensor blob
    given as blob_out. Supports float32 and int32 blob data types. This function
    is intended for copying numpy data into a Caffe2 blob in PythonOps.
    """
    # Some awkward voodoo required by Caffe2 to support int32 blobs
    needs_int32_init = False
    try:
        _ = blob.data.dtype  # noqa
    except Exception:
        needs_int32_init = blob_in.dtype == np.int32
    if needs_int32_init:
        # init can only take a list (failed on tuple)
        blob_out.init(list(blob_in.shape), caffe2_pb2.TensorProto.INT32)
    else:
        blob_out.reshape(blob_in.shape)
    blob_out.data[...] = blob_in


def get_loss_gradients(model, loss_blobs):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_grad = model.net.ConstantFill(b, [b + '_grad'], value=1.0)
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())
