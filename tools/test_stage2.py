#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:03:04 2019

@author: gaomingda
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import pprint
import sys
import os
import random

from caffe2.python import workspace, core

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
import detectron.utils.train

import detectron.core.test_engine as test_engine
from detectron.core.test import im_detect_all
import detectron.core.test_res_top as test_res_top
from collections import defaultdict
from detectron.utils.timer import Timer
import detectron.utils.vis as vis_utils
from detectron.modeling.detector import DetectionModelHelper
import detectron.utils.blob as blob_utils
import detectron.modeling.optimizer as optim
import re
import detectron.utils.net as nu
from detectron.utils.training_stats import TrainingStats
from tensorboardX import SummaryWriter
from detectron.utils import lr_policy

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()
c2_utils.import_custom_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
gpu_id = 0
input_size = [384, 384]
output_dir = "./detectron-output/stage2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
root = "/home/gaomingda/datasets/lip_single"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='Use cfg.NUM_GPUS GPUs for inference',
        action='store_true'
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


class DataLoader(object):
    def __init__(self, data_root, file_ids, cfg, test_model, is_train=False):
        self.is_train = is_train
        if is_train:
            self.data_root = os.path.join(data_root, 'train_images')
            self.label_root = os.path.join(data_root, 'train_segmentations')
        else:
            self.data_root = os.path.join(data_root, 'val_images')
            self.label_root = os.path.join(data_root, 'val_segmentations')
        self.file_ids = os.path.join(data_root, file_ids)
        self.cfg = cfg
        self.batch_size = 1  # cfg.TRAIN.IMS_PER_BATCH

        self.roidb = self.create_roidb()
        self.seq = range(len(self.roidb))
        if self.is_train:  # shuffle
            random.shuffle(self.seq)
        self.cur = 0
        self.num_samples = len(self.roidb)

        self.test_model = test_model

    def create_roidb(self):
        print("creating roidb...")
        roidb = []
        assert os.path.exists(self.file_ids), "{} not exists".format(self.file_ids)
        with open(self.file_ids, 'rb') as f:
            lines = f.readlines()
            ii = 0
            for line in lines:
                print(ii)
                ii += 1
                entry = {}
                img_id = line.strip()
                img_path = os.path.join(self.data_root, img_id + '.jpg')
                #assert os.path.exists(img_path)
                im = cv2.imread(img_path)
                entry['id'] = img_id
                entry['image'] = img_path
                entry['height'] = im.shape[0]
                entry['width'] = im.shape[1]
                entry['label'] = os.path.join(self.label_root, img_id + '.png')
                roidb.append(entry)
        print("created roidb...")
        return roidb

    def next_batch(self):
        data_blob = np.zeros((self.batch_size, input_size[0], input_size[1], 40), dtype=np.float32)
        gt_blob = np.zeros((self.batch_size, input_size[0], input_size[1]), dtype=np.int32)
        batch_id = 0
        # for i in range(self.batch_size):
        meta = []
        while True:
            # print("start loading data batch {}".format(batch_id))
            entry = self.roidb[self.cur]
            self.cur += 1
            if self.cur >= self.num_samples:
                self.cur = 0
                if self.is_train:  # shuffle
                    random.shuffle(self.seq)

            im = cv2.imread(entry['image'])
            label = cv2.imread(entry['label'], 0)
            # model test create human
            timers = defaultdict(Timer)
            with c2_utils.NamedCudaScope(gpu_id):
                cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                    self.test_model, im, None, timers, entry
                )
                mask_res_top = test_res_top.res_top_result(self.test_model, entry, save_png=False)
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            mask_png_20 = vis_utils.vis_one_image(
                im[:, :, ::-1],
                '{:d}_{:s}'.format(self.cur, im_name),
                output_dir,
                cls_boxes_i,
                segms=cls_segms_i,
                keypoints=cls_keyps_i,
                thresh=cfg.VIS_TH,
                box_alpha=0.8,
                dataset=None,
                show_class=False,
                save_png=False
            )
            if len(mask_png_20.shape) == 2:
                print("data next continue")
                continue
            # resize
            # print("mask_res_top shape:", mask_res_top.shape)
            mask_restop_resize = cv2.resize(mask_res_top.transpose([1, 2, 0]), (input_size[0], input_size[1]),
                                            interpolation=cv2.INTER_LINEAR)
            # print("mask_restop_resize: ", mask_restop_resize.shape)
            data_blob[batch_id, :, :, 0:20] = mask_restop_resize
            # print("data_blob:", data_blob.shape)

            # print("mask_png_20 shape", mask_png_20.shape)
            mask_png_20 = mask_png_20.transpose((1, 2, 0))
            data_blob[batch_id, :, :, 20:] = cv2.resize(mask_png_20, (input_size[0], input_size[1]),
                                                        interpolation=cv2.INTER_LINEAR)

            gt_blob[batch_id] = cv2.resize(label, (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)

            meta.append(entry)
            batch_id += 1
            if batch_id >= self.batch_size:
                break
        data_blob = np.transpose(data_blob, [0, 3, 1, 2])  # NHWC-> NCHW
        return data_blob, gt_blob, meta


def create_model(train, cfg, output_dir):
    logger = logging.getLogger(__name__)
    start_iter = 0
    checkpoints = {}
    weights_file = None
    if cfg.TRAIN.AUTO_RESUME:
        # Find the most recent checkpoint (highest iteration number)
        files = os.listdir(output_dir)
        for f in files:
            iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
            if len(iter_string) > 0:
                checkpoint_iter = int(iter_string[0])
                if checkpoint_iter > start_iter:
                    # Start one iteration immediately after the checkpoint iter
                    start_iter = checkpoint_iter + 1
                    resume_weights_file = f

        if start_iter > 0:
            # Override the initialization weights with the found checkpoint
            weights_file = os.path.join(output_dir, resume_weights_file)
            logger.info(
                '========> Resuming from checkpoint {} at start iter {}'.
                    format(weights_file, start_iter)
            )
        # Check for the final model (indicates training already finished)
        final_path = os.path.join(output_dir, 'model_final.pkl')
        if os.path.exists(final_path):
            logger.info('model_final.pkl exists!')
            weights_file = final_path

    logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
    model = DetectionModelHelper(
        name="fusion",
        train=train,
        num_classes=cfg.MODEL.NUM_CLASSES,
        init_params=train
    )
    model.only_build_forward_pass = False
    model.target_gpu_id = gpu_id

    def _single_gpu_build_func(model):
        loss = {"loss": None}
        p = model.Conv("data_stage2", 'conv1_stage2', 40, 20, kernel=3, pad=1, stride=1, no_bias=1,
                       weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}))
        p = model.AffineChannel(p, 'conv1_bn_stage2', dim=20, inplace=True)
        p = model.Relu(p, p)
        human_fc = model.Conv(p, 'conv2_stage2', 20, 20, kernel=1, pad=0, stride=1,
                              weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}))
        if not model.train:
            model.net.NCHW2NHWC(human_fc, 'seg_score_NHWC_stage2')
            model.net.Softmax('seg_score_NHWC_stage2', 'probs_human_NHWC_stage2', axis=3)
            model.net.NHWC2NCHW('probs_human_NHWC_stage2', 'probs_human_NCHW_stage2')
        loss_gradient = None
        if model.train:
            model.net.NCHW2NHWC(human_fc, 'seg_score_NHWC_stage2')
            model.Reshape('seg_score_NHWC_stage2', ['seg_score_reshape_stage2', 'seg_score_old_shape_stage2'],
                          shape=[-1, model.num_classes])
            model.Reshape('gt_label_stage2', ['gt_label_reshape_stage2', 'gt_label_shape_stage2'], shape=[-1, ])

            probs_human, loss_human = model.net.SoftmaxWithLoss(
                ['seg_score_reshape_stage2', 'gt_label_reshape_stage2'],
                ['probs_human_stage2', 'loss_human_stage2'],
                scale=1. / cfg.NUM_GPUS)
            loss_gradient = blob_utils.get_loss_gradients(model, [loss_human])
            model.AddLosses('loss_human_stage2')
        loss['loss'] = loss_gradient

        if model.train:
            loss_gradients = {}
            for lg in loss.values():
                if lg is not None:
                    loss_gradients.update(lg)
            return loss_gradients
        else:
            return None

    optim.build_data_parallel_model(model, _single_gpu_build_func)

    # Performs random weight initialization as defined by the model
    workspace.RunNetOnce(model.param_init_net)
    return model, weights_file, start_iter, checkpoints


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, 'net.pbtxt'), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(os.path.join(output_dir, 'param_init_net.pbtxt'), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))


def handle_critical_error(model, msg):
    logger = logging.getLogger(__name__)
    logger.critical(msg)
    model.roi_data_loader.shutdown()
    raise Exception(msg)


def main():
    # Initialize C2
    workspace.GlobalInit(
        ['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1']
    )
    # Set up logging and load config options
    logger = setup_logging(__name__)
    logging.getLogger('detectron.roi_data.loader').setLevel(logging.INFO)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    smi_output, cuda_ver, cudnn_ver = c2_utils.get_nvidia_info()
    logger.info("cuda version : {}".format(cuda_ver))
    logger.info("cudnn version: {}".format(cudnn_ver))
    logger.info("nvidia-smi output:\n{}".format(smi_output))
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    # Note that while we set the numpy random seed network training will not be
    # deterministic in general. There are sources of non-determinism that cannot
    # be removed with a reasonble execution-speed tradeoff (such as certain
    # non-deterministic cudnn functions).
    np.random.seed(cfg.RNG_SEED)
    # test model
    logger.info("creat test model ...")
    test_model = test_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS, gpu_id=0)
    logger.info("created test model ...")
    #cfg.TRAIN.IMS_PER_BATCH = 1
    train_data = DataLoader(root, "val_id.txt", cfg, test_model, is_train=False)
    # creat mode
    model, weights_file, start_iter, checkpoints = create_model(False, cfg, output_dir)
    # test blob
    print(workspace.Blobs())
    # create input blob
    blob_names = ['data_stage2']
    for gpu_id in range(cfg.NUM_GPUS):
        with c2_utils.NamedCudaScope(gpu_id):
            for blob_name in blob_names:
                workspace.CreateBlob(core.ScopedName(blob_name))
    # Override random weight initialization with weights from a saved model
    if weights_file:
        nu.initialize_gpu_from_weights_file(model, weights_file, gpu_id=0)
    # Even if we're randomly initializing we still need to synchronize
    # parameters across GPUs
    nu.broadcast_parameters(model)
    workspace.CreateNet(model.net)

    logger.info('Outputs saved to: {:s}'.format(os.path.abspath(output_dir)))

    logger.info("start test ...")
    save_root = os.path.join(output_dir, 'fusion')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for cur_iter in range(10000):
        # feed data
        # print("{} iter starting feed data...".format(cur_iter))
        data_stage2, gt_label, meta = train_data.next_batch()
        '''# 
        print('input0-20 sungalsses max score:', np.max(data_stage2[0, 4, :, :]))
        print('input20-40 sungalsses max score:', np.max(data_stage2[0, 24, :, :]))
        print('input0-20 glovess max score:', np.max(data_stage2[0, 3, :, :]))
        print('input20-40 glovess max score:', np.max(data_stage2[0, 23, :, :]))
        #'''
        with c2_utils.NamedCudaScope(gpu_id):
            workspace.FeedBlob(core.ScopedName('data_stage2'), data_stage2)

        # print("workspace.RunNet(model.net.Proto().name)")
        with c2_utils.NamedCudaScope(gpu_id):
            workspace.RunNet(model.net.Proto().name)
            batch_probs = workspace.FetchBlob(core.ScopedName('probs_human_NCHW_stage2'))
            batch_probs = batch_probs.transpose((0, 2, 3, 1))
        assert len(meta) == batch_probs.shape[0]
        #print('batch_probs shape:', batch_probs.shape)
        for i in range(len(meta)):
            probs = cv2.resize(batch_probs[i], (meta[i]['width'], meta[i]['height']), interpolation=cv2.INTER_LINEAR)
            probs = probs.transpose((2,0,1))
            print('sungalsses max score:', np.max(probs[4, :, :]))
            print('glovess max score:', np.max(probs[3, :, :]))
            #print('probs shape:', probs.shape)
            cv2.imwrite(os.path.join(save_root, meta[i]['id']+'.png'), probs.argmax(0))
        print("prossed ", cur_iter)



def test_model(model_file, multi_gpu_testing, opts=None):
    """Test a model."""
    # Clear memory before inference
    workspace.ResetWorkspace()
    # Run inference
    run_inference(
        model_file, multi_gpu_testing=multi_gpu_testing,
        check_expected_results=True,
    )


if __name__ == '__main__':
    main()
