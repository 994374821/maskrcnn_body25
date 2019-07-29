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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
from PIL import Image
import cv2
# Must happen before importing COCO API (which imports matplotlib)
import detectron.utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from detectron.core.config import cfg
from detectron.utils.timer import Timer
import detectron.datasets.dataset_catalog as dataset_catalog
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils
from detectron.datasets.dataset_catalog_LIP import ANN_FN
from detectron.datasets.dataset_catalog_LIP import DATASETS
from detectron.datasets.dataset_catalog_LIP import IM_DIR
from detectron.datasets.dataset_catalog_LIP import IM_IDS
from detectron.datasets.dataset_catalog_LIP import IM_PREFIX

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        if 'train' in name:
            assert os.path.exists(DATASETS[name][ANN_FN]), \
                'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        assert os.path.exists(DATASETS[name][IM_IDS]), \
            'im_ids file \'{}\' not found'.format(DATASETS[name][IM_IDS])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.dataset = name.split('_')[-1] # 'train' or 'val'
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        #self.COCO = COCO(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        # Set up dataset classes
        #category_ids = self.COCO.getCatIds()
        if 'ATR' in self.name:
            category_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            categories = ['background', 'hat', 'hair', 'sunglasses', 'upperclothes',
	               'skirt', 'pants', 'dress', 'belt', 'leftShoes', 'right-shoe', 'face',
	               'left-leg', 'right-leg', 'left-arm', 'right-arm', 'bag', 'scarf']
        else:
            category_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            #categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
            categories = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
	               'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
	               'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe','rightShoe']
        if cfg.Ignore_left: # 14,15, 16,17, 18,19
            if 'ATR' in self.name:
                categories = ['background', 'hat', 'hair', 'sunglasses', 'upperclothes',
	               'skirt', 'pants', 'dress', 'belt', 'shoe', 'face', 'leg', 'arm', 'bag', 'scarf']
                category_ids = range(len(categories))
                self.category_id_to_Ignore_left_id = {
                        v: i
                        for i, v in enumerate(range(18))
                        }
                self.category_id_to_Ignore_left_id[10] = 9
                self.category_id_to_Ignore_left_id[11] = 10
                self.category_id_to_Ignore_left_id[12] = 11
                self.category_id_to_Ignore_left_id[13] = 11
                self.category_id_to_Ignore_left_id[14] = 12
                self.category_id_to_Ignore_left_id[15] = 12
                self.category_id_to_Ignore_left_id[16] = 13
                self.category_id_to_Ignore_left_id[17] = 14
            else:
                categories = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
	               'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
	               'face', 'Arm', 'Leg', 'Shoe']
                category_ids = range(len(categories))
                self.category_id_to_Ignore_left_id = {
                        v: i
                        for i, v in enumerate(range(20))
                        }
                self.category_id_to_Ignore_left_id[15] = 14
                self.category_id_to_Ignore_left_id[16] = 15
                self.category_id_to_Ignore_left_id[17] = 15
                self.category_id_to_Ignore_left_id[18] = 16
                self.category_id_to_Ignore_left_id[19] = 16
            
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = categories
        self.num_classes = len(self.classes)
        logger.info('classes: {}'.format(self.classes))
        logger.info('num_classes: {}'.format(self.num_classes))
        self.json_category_id_to_contiguous_id = {
            v: i 
            for i, v in enumerate(category_ids)
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()

    def get_roidb(
        self,
        gt=False,
        proposal_file=None,
        min_proposal_size=2,
        proposal_limit=-1,
        crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        roidb = self._load_lip() # load data when train or test   
        if gt:
            # include gt object annotations
            self.debug_timer.tic()
            self.load_lip_annotations(roidb)
            logger.debug(
                 'load_lip_annotations took {:.3f}s'.
                 format(self.debug_timer.toc(average=False))
                 )
        
        #############################################
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        return roidb
    
    def _load_lip(self):
        """ gao: load train or test dadaset of LIP"""
        imglist_file = DATASETS[self.name][IM_IDS]
        assert os.path.exists(imglist_file), 'path does not exist: {}'.format(imglist_file)
        imgids_list = []
        with open(imglist_file) as f:
            for line in f.readlines():
                if len(line)>1:
                    imgids_list.append(line.strip())
        # mistake label id
        
        if 'LIP_train' in self.name: 
            mistakelist_file = os.path.join(os.path.dirname(imglist_file), 'train_mistake_id.txt')
            assert os.path.exists(mistakelist_file), 'path does not exist: {}'.format(mistakelist_file)
            im_mistake_ids = []
            with open(mistakelist_file) as f:
                for line in f.readlines():
                    if len(line)>1:
                        im_mistake_ids.append(line.strip())

        roidb = []
        for i in range(len(imgids_list)):
            if 'LIP_train' in self.name:
                if imgids_list[i] in im_mistake_ids:
                    continue
            roi_entry = dict()
            roi_entry['dataset'] = self
            roi_entry['id'] = imgids_list[i]
            roi_entry['image'] = os.path.join(DATASETS[self.name][IM_DIR], imgids_list[i] + '.jpg')
            assert os.path.exists(roi_entry['image']), 'image path does not exist: {}'.format(roi_entry['images'])
            
            img = cv2.imread(roi_entry['image'])
            size = img.shape
            roi_entry['height'] = size[0]
            roi_entry['width'] = size[1]
            
            roi_entry['flipped'] = False
            roi_entry['has_visible_keypoints'] = False
            roi_entry['boxes'] = np.empty((0,4), dtype=np.float32)
            roi_entry['gt_classes'] = np.empty((0), dtype=np.int32)
            roi_entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
            roi_entry['gt_overlaps'] = scipy.sparse.csr_matrix(
                np.empty((0,self.num_classes), dtype=np.float32)
                )
            roi_entry['is_crowd'] = np.empty((0), dtype=np.bool)
            roi_entry['seg_areas'] = np.empty((0), dtype=np.float32)
            
            roidb.append(roi_entry)
        return roidb
        
    def load_lip_annotations(self,roidb):        
            # load from label of png
        for i in range(len(roidb)):
            roi_entry = roidb[i]
            if roi_entry['id'] in ['27969_199668']:
                continue
            #print(i, roi_entry['id'])
            boxes, gt_classes, ins_id, label_path, gt_overlaps = self.load_from_seg(roi_entry['id'])
            if boxes.size == 0:
                total_num_objs = 0
                boxes = np.zeros((total_num_objs, 4), dtype=np.uint16)
                gt_overlaps = np.zeros((total_num_objs, self.num_classes), dtype=np.float32)
                gt_classes = np.zeros((total_num_objs, ), dtype=np.int32)
            roi_entry['boxes'] = boxes
            roi_entry['gt_classes'] = gt_classes
            roi_entry['box_to_gt_ind_map'] = ins_id 
            roi_entry['ins_seg'] = label_path # full path of label png
           # im_label = Image.open(label_path)
           # pixel = list(im_label.getdata())
           # im_label = np.array(pixel).reshape([im_label.size[1], im_label.size[0]])
           # roi_entry['ins_seg'] = im_label
            
            roi_entry['gt_overlaps'] = gt_overlaps
            roi_entry['gt_overlaps'] = scipy.sparse.csr_matrix(roi_entry['gt_overlaps'])
            #roi_entry['max_overlaps'] = gt_overlaps.max(axis=1)
            #roi_entry['max_class'] = gt_overlaps.argmax(axis=1)            
            roi_entry['is_crowd'] = np.zeros((boxes.shape[0]), dtype=np.bool)
            #roi_entry['has_visible_keypoints'] = False
            roi_entry['seg_areas'] = np.zeros((boxes.shape[0]), dtype=np.float32)
            roi_entry['seg_areas'][:] = 50
            #roi_entry['gt_boxes'] = boxes
            #roidb.append(roi_entry)
        
        #return roidb			
	
            
    def load_from_seg(self,seg_gt_id):
        """ gao: load from seg label png """
        seg_gt = os.path.join(DATASETS[self.name][ANN_FN], seg_gt_id + '.png')
        assert os.path.exists(seg_gt), 'path does not exist: {}'.format(seg_gt)
        im = Image.open(seg_gt)
        pixel = list(im.getdata())
        pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
        gt_classes = []
        boxes = []
        box_to_gt_ind_map = []
        gt_overlaps = []
        ins_id = 0
        for c in range(1,self.num_classes):
            px = np.where(pixel == c)
            if len(px[0])==0:
                continue
            x_min = np.min(px[1])
            y_min = np.min(px[0])
            x_max = np.max(px[1])
            y_max = np.max(px[0])
            if x_max - x_min <= 1 or y_max - y_min <= 1:
                continue
            
            if cfg.Ignore_left:
                c = self.category_id_to_Ignore_left_id[c]                    
#            gt_classes.append(c)
#            boxes.append([x_min, y_min, x_max, y_max])
#            box_to_gt_ind_map.append(ins_id)
#            ins_id += 1
#            overlaps = np.zeros(self.num_classes,dtype=np.float32)
#            overlaps[c] = 1
#            gt_overlaps.append(overlaps)
            if (c==3 or c==8) and 'LIP' in cfg.TRAIN.DATASETS[0]: # has gloves or socks            
                box,gt_class,box_to_gt_ind,gt_overlap,ins_id = _get_socks_glove(pixel,c,ins_id,self.num_classes)
                for i in range(len(box)):
                    boxes.append(box[i])
                    gt_classes.append(gt_class[i])
                    box_to_gt_ind_map.append(box_to_gt_ind[i])
                    gt_overlaps.append(gt_overlap[i])
            else:                
                gt_classes.append(c)
                boxes.append([x_min, y_min, x_max, y_max])
                box_to_gt_ind_map.append(ins_id)
                ins_id += 1
                overlaps = np.zeros(self.num_classes)
                overlaps[c] = 1
                gt_overlaps.append(overlaps)
        
        return np.asarray(boxes, dtype=np.float32), np.asarray(gt_classes,dtype=np.int32), np.asarray(box_to_gt_ind_map,dtype=np.int32), seg_gt, np.asarray(gt_overlaps)

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]

def _get_socks_glove(label,c,ins_id,num_classes):
    """ gao: get gt annotations of socks and glove"""
    px = np.where(label == c)
    x = np.unique(px[1]) # x coordinate
    y = np.unique(px[0])
    distance1 = 10
    distance2 = 5
    flag = 0
    x_flag = 0
    #y_flag = 0
    if (x[-3]-x[2])>(len(x)+distance1):                    
        start = 2
        end = len(x)-4
        x_flag = 1
        while flag==0:
            md = int((end+start)/2)
            if (x[md]-x[start])>(md-start+distance2):                
                if (x[md]-x[md-1])>distance2:
                    flag=md - 1
                elif (x[md+1]-x[md])>distance2:
                    flag = md
                else:
                    end = md
                    md = int((end+start)/2)
            elif (x[end]-x[md])>(end-md+distance2):
                if (x[md]-x[md-1])>distance2:
                    flag=md - 1
                elif (x[md+1]-x[md])>distance2:
                    flag = md
                else:
                    start = md
                    md = int((end+start)/2)
            if (end-start)<3:
                break
    elif (y[-3]-y[2])>(len(y)+distance1):
        start = 2
        end = len(y)-4
        #y_flag = 1
        while flag==0:
            md = int((end+start)/2)
            if (y[md]-y[start])>(md-start+distance2):
                if (y[md]-y[md-1])>distance2:
                    flag = md -1
                elif (y[md+1]-y[md])>distance2:
                    flag = md 
                else:
                    end = md
                    md = int((end+start)/2)
            elif (y[end]-y[md])>(end-md+distance2):
                if (y[md]-y[md-1])>distance2:
                    flag = md -1
                elif (y[md+1]-y[md])>distance2:
                    flag = md 
                else:
                    start = md
                    md = int((end+start)/2)
            if (end-start)<3:
                break
    gt_classes = []
    boxes = []
    box_to_gt_ind_map = []
    gt_overlaps = []
    overlaps = np.zeros(num_classes)
    overlaps[c] = 1
    if flag!=0:
        if x_flag==1:
            y_0 = np.where(label[:, x[0]:x[flag]]==c)[0]
            y_0.sort()
            boxes.append([x[0],y_0[0],x[flag],y_0[-1]])
            
            y_1 = np.where(label[:, x[flag+1]:x[-1]]==c)[0]
            y_1.sort()
            boxes.append([x[flag+1],y_1[0],x[-1],y_1[-1]])
        else:
            x_0 = np.where(label[ y[0]:y[flag], :]==c)[1]
            x_0.sort()
            boxes.append([x_0[0],y[0],x_0[-1],y[flag]])
            
            x_1 = np.where(label[y[flag+1]:y[-1],:]==c)[1]
            x_1.sort()
            boxes.append([x_1[0],y[flag+1],x_1[-1],y[-1]])            
        gt_classes.append(c)
        gt_classes.append(c)
        box_to_gt_ind_map.append(ins_id)
        box_to_gt_ind_map.append(ins_id+1)        
        ins_id += 2        
        gt_overlaps.append(overlaps)
        gt_overlaps.append(overlaps)
    else:
        boxes.append([x[0],y[0],x[-1],y[-1]])
        gt_classes.append(c)
        box_to_gt_ind_map.append(ins_id)
        ins_id += 1
        gt_overlaps.append(overlaps)
        
    return boxes,gt_classes,box_to_gt_ind_map,gt_overlaps,ins_id
