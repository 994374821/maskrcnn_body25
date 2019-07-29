#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:53:23 2019

@author: gaomingda
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.rpn_generator import generate_rpn_on_dataset
from detectron.core.rpn_generator import generate_rpn_on_range
from detectron.core.test import im_detect_all
from detectron.datasets import task_evaluation
from detectron.datasets.json_dataset import JsonDataset
from detectron.modeling import model_builder
from detectron.utils.io import save_object
from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.net as net_utils
import detectron.utils.subprocess as subprocess_utils
import detectron.utils.vis as vis_utils

import detectron.core.test_engine as test_engine
from detectron.core.config import merge_cfg_from_file
#import sys
#sys.path.append('/home/gaomingda/softer_nms_LIP_JPPNet/detectron/LIP_JPPNet')
#from evaluate_pose_JPPNet import generate_poseJPPNet_pred_model
import detectron.core.test_res_top as test_res_top

from tkinter import *
from tkinter import filedialog
import tkinter
from PIL import Image, ImageTk
import matplotlib

import pprint

from detectron.core.config import assert_and_infer_cfg

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)

root_path = os.path.dirname(os.path.realpath(__file__))

File = None
cfg_file = os.path.join(root_path, 'demo', 'e2e_mask_rcnn_R-50-FPN_test.yaml')
weights_file = os.path.join(root_path, 'demo', 'model_final_res50_lip.pkl')
gpu_id = 0

cache_file= os.path.join(root_path, 'demo', 'cache.png')

colormap = [0, 0, 0, 128, 0, 0, 
       255, 0, 0,
       0, 128, 0,
       165, 42, 42,
       255, 165, 0,
       0, 0, 139,
       30, 144, 255,
       85, 107, 47,
       0, 128, 128,
       199, 21, 133,
       218, 112, 214,
       128, 0, 128,
       0, 0, 255,
       70, 130, 180,
       0, 255, 255,
       72, 209, 204,
       0, 250, 154,
       255, 255, 0,
       255, 215, 0]


def getCmap():
    labelCount = 20
    cmapGen = matplotlib.cm.get_cmap('jet', labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]
    cmap = (cmap * 255).astype(int)
    padding = np.zeros((256-cmap.shape[0], 3), np.int8)
    cmap = np.vstack((cmap, padding))
    cmap = cmap.reshape((-1))
    cmap[0:3] = 0
    return cmap

#cmap = getCmap()
#print(cmap[0:20])
 
if __name__ == "__main__":
    root = Tk()
    root.title('human parsing')
    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    
    canvas = Canvas(frame,width=340,height=305)
    canvas.grid(row=0,column=0)
    
    canvas2 = Canvas(frame,width=340,height=100)
    canvas2.grid(row=1,column=0)
    
    frame.pack(fill=BOTH,expand=1)
    
    # model init
    merge_cfg_from_file(cfg_file)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))
        
    model = test_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS, gpu_id=gpu_id)
    
    cmap = getCmap()
    cmap[0:60] = colormap
    
    # color map png
    color_png = Image.open(os.path.join(root_path, 'demo', 'color', 'color_map.jpg')).resize((340,100))
    filename = ImageTk.PhotoImage(color_png)
    canvas2.image = filename  # <--- keep reference of your image
    canvas2.create_image(0,0,anchor='nw',image=filename)
    
    #function to be called when mouse is clicked
    def printcoords():
        global File
        global filename
        global show_img
        File = filedialog.askopenfilename(parent=root, initialdir=os.path.join(root_path, 'demo', 'lip_val'),title='Choose an image.')
        show_img = Image.open(File).resize((160,300))
        filename = ImageTk.PhotoImage(show_img)
#        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(0,0,anchor='nw',image=filename)

    def detect():     
        im = cv2.imread(File)
        entry = {}
        entry['id'] = File.split('/')[-1].split('.')[0]
        entry['image'] = File
        print(entry)
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(gpu_id):
            cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                model, im, None, timers, entry
            )
            mask_res_top = test_res_top.res_top_result(model, entry)
        
        resize_pred = cv2.resize(mask_res_top.argmax(0), (160, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(cache_file, resize_pred)  
        
        global png
        global png_pred
        png = Image.open(cache_file).convert('P')
        png.putpalette(cmap)
        png_pred = ImageTk.PhotoImage(png)
#        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(170,0,anchor='nw',image=png_pred)
        print('done')

    b = Button(root,anchor='s', text='choose',command=printcoords).pack(side=LEFT, padx=70)
    b=Button(root,anchor='s', text='detect', command=detect).pack(side=LEFT)

#    print(File)
    root.mainloop()

