#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 05:43:09 2018

@author: gaomingda
"""

import os
import numpy as np
from PIL import Image


def main():
	image_paths, label_paths = init_path()
	hist = compute_hist(image_paths, label_paths)
	show_result(hist)

def init_path():
    image_dir = '/home/gaomingda/Documents/maskrcnn_fcn/detectron-output/test/ATR_val/generalized_rcnn/fusion'
#    image_dir = '/home/gaomingda/detectron_LIP/detectron-output/test/LIP_val/generalized_rcnn/vis_7_06/'
#    image_dir = '/home/gaomingda/Documents/LIP_result/test'
    label_dir = '/home/gaomingda/Downloads/gaomingda/dataset/LIP/humanparsing/segmentations'
    file_names = os.listdir(image_dir)
    print('number predict image:',len(file_names))
    image_paths = []
    label_paths = []
    for file_name in file_names:
        if '.png' in file_name:
            image_paths.append(os.path.join(image_dir, file_name))
            label_paths.append(os.path.join(label_dir, file_name))
    return image_paths, label_paths

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(images, labels):
    n_cl = 18
    
    hist = np.zeros((n_cl, n_cl))
    for img_path, label_path in zip(images, labels):
        assert os.path.exists(img_path), "{} do not exist".format(img_path)
        assert os.path.exists(label_path), "{} do not exist".format(label_path)
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)
        
        gtsz = label_array.shape
        imgsz = image_array.shape
        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)
            
        hist += fast_hist(label_array, image_array, n_cl)
        
    return hist
    

def show_result(hist):

	classes = ['background', 'hat', 'hair', 'sunglasses', 'upperclothes',
	               'skirt', 'pants', 'dress', 'belt', 'leftShoes', 'right-shoe', 'face',
	               'left-leg', 'right-leg', 'left-arm', 'right-arm', 'bag', 'scarf']
	# num of correct pixels
	num_cor_pix = np.diag(hist)
	# num of gt pixels
	num_gt_pix = hist.sum(1)
	print '=' * 50

	# @evaluation 1: overall accuracy
	acc = num_cor_pix.sum() / hist.sum()
	print '>>>', 'overall accuracy', acc
	print '-' * 50

	# @evaluation 2: mean accuracy & per-class accuracy 
	print 'Accuracy for each class (pixel accuracy):'
	for i in xrange(len(classes)):
		print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i])) 
	acc = num_cor_pix / num_gt_pix
	print '>>>', 'mean accuracy', np.nanmean(acc)
	print '-' * 50
	
	# @evaluation 3: mean IU & per-class IU
	union = num_gt_pix + hist.sum(0) - num_cor_pix
	for i in xrange(len(classes)):
		print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
	iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
	print '>>>', 'mean IU', np.nanmean(iu)
	print '-' * 50

	# @evaluation 4: frequency weighted IU
	freq = num_gt_pix / hist.sum()
	print '>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()
	print '=' * 50



if __name__ == '__main__':
    main()
