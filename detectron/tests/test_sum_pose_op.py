#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 23:09:37 2018

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
from caffe2.python import workspace

import detectron.utils.c2 as c2_utils


class ZeroEvenOpTest(unittest.TestCase):

    def _run_zero_even_op(self, X, Cout):
        op = core.CreateOperator('SumPose', ['X'], ['Y'], Cout=Cout)
        workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        print('Y shape:', Y.shape)
        return Y
    def _run_zero_even_op_gpu(self, X, Cout):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('SumPose', ['X'], ['Y'], Cout=Cout)
            workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y
    def test_preserves_vals_at_odd_inds(self):
        np.random.seed(1)
        X = np.random.rand(4, 16, 8, 8).astype(np.float32) # (n, c, h, w)
        Y_exp = self.calulate_Y(X, 10)
        Y_act = self._run_zero_even_op(X, 10)
#        print('Y_exp shape:', Y_exp.shape)
#        print('Y_exp:', Y_exp)
#        print('Y_act:', Y_act)
#        print(Y_exp==Y_act)
        np.testing.assert_allclose(Y_act, Y_exp)
        
        Y_act_cuda = self._run_zero_even_op_gpu(X, 10)
        np.testing.assert_allclose(Y_act_cuda, Y_exp)

    def calulate_Y(self, X, cout):
        n, c, h, w = X.shape
        Y = np.zeros((n, cout, h, w), dtype=np.float32)
        sum_onehot = np.zeros((n, h, w), dtype=np.float32)
        for i in range(c):
            sum_onehot[:, :, :] += X[:, i, :, :]
        for j in range(cout):
            Y[:, j, :, :] = sum_onehot
        #test
        for n_ in range(n):
            sum_n = np.zeros((h, w), dtype=np.float32)
            for i in range(c):
                sum_n += X[n_, i, :, :]
            for j in range(cout):
                print((Y[n_, j, :, :]==sum_n).all())
        return Y
    
def calulate_Y( X, cout):
    n, c, h, w = X.shape
    Y = np.zeros((n, cout, h, w), dtype=np.float32)
    sum_onehot = np.zeros((n, h, w), dtype=np.float32)
    for i in range(c):
        sum_onehot[:, :, :] += X[:, i, :, :]
    for j in range(cout):
        Y[:, j, :, :] = sum_onehot
    return Y

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    c2_utils.import_custom_ops()
    assert 'SumPose' in workspace.RegisteredOperators()
    unittest.main()
   
    