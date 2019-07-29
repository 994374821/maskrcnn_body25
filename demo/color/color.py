#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:19:58 2019

@author: gaomingda
"""

from PIL import Image, ImageTk
import matplotlib

import numpy as np
import os

root = os.path.dirname(os.path.realpath(__file__)) 

rgb = [0, 0, 0, 128, 0, 0, 
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

cmap = getCmap()
cmap[0:60] = rgb

for i in range(20):
    color = np.ones((15,15), dtype=np.int32) * i 
    color = Image.fromarray(color).convert('P')
    color.putpalette(cmap)
    color.save(os.path.join(root, str(i)+'.png'))