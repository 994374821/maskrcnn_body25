#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:53:40 2018

@author: gaomingda
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import gradient_checker
from caffe2.python import workspace

import detectron.utils.c2 as c2_utils
import detectron.utils.logging as logging_utils

class SsDistanceLossOpTest(unittest.TestCase):
#class SsDistanceLossOpTest(object):
    def _run_test(self, scores, labels, rois, bbox_pred, gt_centers, im_info,  
                  regression_ws, num_classes, check_grad=False):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
#            op = core.CreateOperator('SsDistanceLoss', 
#                                     ['scores', 'labels', 'rois', 'bbox_pred', 'gt_centers', 'im_info'], 
#                                     ['loss', 'p', 'pred_class'],
#                                     regression_ws=regression_ws, num_classes=num_classes)
            op = core.CreateOperator('SsDistanceLoss', 
                                     ['scores', 'labels', 'rois', 'bbox_pred', 'gt_centers', 'im_info'], 
                                     ['loss', 'prob', 'pred_class'],
                                     num_classes=num_classes)
            workspace.FeedBlob('scores', scores)
            workspace.FeedBlob('labels', labels)
            workspace.FeedBlob('rois', rois)
            workspace.FeedBlob('bbox_pred', bbox_pred)
            workspace.FeedBlob('gt_centers', gt_centers)
            workspace.FeedBlob('im_info', im_info)
        workspace.RunOperatorOnce(op)
        pred_class = workspace.FetchBlob('pred_class')
        prob = workspace.FetchBlob('prob')
        loss = workspace.FetchBlob('loss')
        
        print('pred_class:', pred_class)
        print('loss:', loss)

        if check_grad:
            gc = gradient_checker.GradientChecker(
                stepsize=0.005,
                threshold=0.005,
                device_option=core.DeviceOption(caffe2_pb2.CUDA, 0)
            )

            res, grad, grad_estimated = gc.CheckSimple(op, [scores, labels, rois, bbox_pred, gt_centers, im_info],
                                                       0, [0])
            print('res:', res)
            self.assertTrue(res, 'Grad check failed')

            self.assertTrue(
                grad.shape == grad_estimated.shape,
                'Fail check: grad.shape != grad_estimated.shape'
                )

        
#        print('C_ref:', C_ref)
#        np.testing.assert_allclose(C, C_ref, rtol=1e-5, atol=1e-08)

    def test_small_forward_and_gradient(self):
        # scores
        np.random.seed(0)
        scores = np.random.rand(4, 20).astype(np.float32)
        scores = scores-0.5
        scores[0, 15] = 1
        scores[1, 16] = 1
        scores[2, 19] = 1
        scores[3, 18] = 1
        pred_classes = scores.argmax(axis=1)
        print('scores.argmax(axis=1)', scores.argmax(axis=1))
        print(type(scores))
        scores = scores.reshape((1,-1, 1, 1))
        print('scores.shape:', scores.shape)
        
        labels = np.array([14, 17, 18, 18], dtype=np.int32)
    
        rois = np.array([[0, 2,2,10,10], [0, 20, 20, 30,30], [0, 20, 40, 30, 50], [0, 40, 40, 50, 50]], 
                        dtype=np.float32)
        gt_boxes = rois
        gt_centers = np.ones((1, 12), dtype=np.float32) * -1
        for i in range(labels.shape[0]):
            _, x1, y1, x2, y2 = gt_boxes[i]
            x_c = x1 + (x2 - x1) / 2
            y_c = y1 + (y2 - y1) / 2
            gt_centers[0, (labels[i]-14)*2] = x_c
            gt_centers[0, (labels[i]-14)*2 + 1] = y_c
        print('gt_centers:', gt_centers)
        labels = labels.reshape(1, -1, 1, 1)
        print('labels.shape:', labels.shape)
        rois = rois.reshape(1, -1, 1, 1)
        gt_centers = gt_centers.reshape(1, -1, 1, 1)
        
        im_info = np.array([[60, 60, 1]], dtype=np.float32)
        im_info = im_info.reshape(1, -1, 1, 1)
        bbox_pred = np.zeros((4, 80), dtype=np.float32)
        bbox_pred = bbox_pred.reshape(1, -1, 1, 1)
        
        # args: regression_ws, num_classes, 
        regression_ws = np.array([1, 1, 1, 1], dtype=np.float32)
        num_classes = 20
        
        test(scores, pred_classes, labels, gt_centers, rois, im_info)
        
        self._run_test(scores, labels, rois, bbox_pred, gt_centers, im_info,  
                  regression_ws, num_classes, check_grad=True)
        
        
        
    def test_middle_forward_and_gradient(self):
        # scores
        np.random.seed(0)
        scores = np.random.rand(6, 20).astype(np.float32)
        scores = scores-0.5
        scores[0, 15] = 1
        scores[1, 16] = 1
        scores[2, 19] = 1
        scores[3, 18] = 1
        scores[4, 14] = 1
        scores[5, 13] = 1
        pred_classes = scores.argmax(axis=1)
        print('scores.argmax(axis=1)', scores.argmax(axis=1))
        print(type(scores))
        scores = scores.reshape((1,-1, 1, 1))
        print('scores.shape:', scores.shape)
        
        labels = np.array([14, 16, 18,19, 15, 13], dtype=np.int32)
    
        rois = np.array([[0, 2,2,10,10], [0, 20, 20, 30,30], [0, 20, 40, 30, 50], [0, 40, 40, 50, 50], 
                         [1, 20, 40, 30, 50], [1, 40, 40, 50, 50]], 
                        dtype=np.float32)
        gt_boxes = rois
        gt_centers = np.ones((2, 12), dtype=np.float32) * -1
        for i in range(labels.shape[0]):
            if labels[i]<14:
                continue
            indx_image, x1, y1, x2, y2 = gt_boxes[i]
            x_c = x1 + (x2 - x1) / 2
            y_c = y1 + (y2 - y1) / 2
            gt_centers[int(indx_image), (labels[i]-14)*2] = x_c
            gt_centers[int(indx_image), (labels[i]-14)*2 + 1] = y_c
        print('gt_centers:', gt_centers)
        labels = labels.reshape(1, -1, 1, 1)
        print('labels.shape:', labels.shape)
        rois = rois.reshape(1, -1, 1, 1)
        gt_centers = gt_centers.reshape(1, -1, 1, 1)
        
        im_info = np.array([[60, 60, 1]], dtype=np.float32)
        im_info = im_info.reshape(1, -1, 1, 1)
        bbox_pred = np.zeros((6, 80), dtype=np.float32)
        bbox_pred = bbox_pred.reshape(1, -1, 1, 1)
        
        # args: regression_ws, num_classes, 
        regression_ws = np.array([1, 1, 1, 1], dtype=np.float32)
        num_classes = 20
        
        self._run_test(scores, labels, rois, bbox_pred, gt_centers, im_info,  
                  regression_ws, num_classes, check_grad=True)
        
#        test(scores, pred_classes, labels, gt_centers, rois, im_info)
        
def test(scores, pred_classes, gt_labels, gt_centers, rois, im_info):
    """scores: (1, num_anchors*num_classes, 1, 1)
        pred_classes: (num_ancors,)
        labels: (1, num_anchors, 1, 1)
        gt_centers: (1, 12, 1, 1); one image
        rois: (1, 5*num_anchors, 1, 1)
        im_info: (1, 3*num_images, 1, 1)
    """
    def sigmoid(x):
        return 1./(1+ np.exp(-x))
    loss = 0
    for i in range(len(pred_classes)):
        pred_class = pred_classes[i]
        if pred_class<14:
            continue
        label = gt_labels[0, i, 0, 0]
        score_index = i * 20 + pred_class
        
#        print(label, type(label), label.shape)
        if label==pred_class:
            loss += -np.log(sigmoid(scores[0, score_index, 0, 0]))
        else:
            dis = 1
            gt_center_index = (pred_class-14) * 2
            if gt_centers[0, gt_center_index, 0, 0]!=-1:
                gt_cx = gt_centers[0, gt_center_index, 0, 0]
                gt_cy = gt_centers[0, gt_center_index+1, 0, 0]
                
                anchor_x1, anchor_y1, anchor_x2, anchor_y2 = rois[0, i*5+1: i*5+5, 0, 0]
                anchor_cx = anchor_x1 + (anchor_x2 - anchor_x1)/2
                anchor_cy = anchor_y1 + (anchor_y2 - anchor_y1)/2
                im_w = im_info[0, 0, 0, 0]
                im_h = im_info[0, 1, 0, 0]
                dis = np.sqrt( (gt_cx-anchor_cx)**2 + (gt_cy-anchor_cy)**2 )/ np.sqrt(im_w**2 + im_h**2)
            loss += -np.log(1-sigmoid(scores[0, score_index, 0, 0])) * dis
    print('loss:', loss)
                
       
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    c2_utils.import_custom_ops()
    assert 'SsDistanceLoss' in workspace.RegisteredOperators()
    unittest.main()
#    ss_test = SsDistanceLossOpTest()
#    ss_test.test_small_forward_and_gradient()
    
    