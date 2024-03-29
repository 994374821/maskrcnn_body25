#!/usr/bin/env python2

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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import numpy as np
import json

from caffe2.python import utils as c2_py_utils

from detectron.core.config import cfg
from detectron.utils.logging import log_json_stats
from detectron.utils.logging import SmoothedValue
from detectron.utils.timer import Timer
import detectron.utils.net as nu


class TrainingStats(object):
    """Track vital training statistics."""

    def __init__(self, model, writer):
        # Window size for smoothing tracked values (with median filtering)
        self.WIN_SZ = 20
        # Output logging period in SGD iterations
        self.LOG_PERIOD = 20
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(self.WIN_SZ)
            for key in model.losses + model.metrics
        }
        self.losses_and_metrics = {
            key: 0
            for key in model.losses + model.metrics
        }
        self.smoothed_total_loss = SmoothedValue(self.WIN_SZ)
        self.smoothed_mb_qsize = SmoothedValue(self.WIN_SZ)
        self.iter_total_loss = np.nan
        self.iter_timer = Timer()
        self.model = model
        self.writer = writer

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self, i_iter=0):
        """Update tracked iteration statistics."""
        for k in self.losses_and_metrics.keys():
            if k in self.model.losses:
                self.losses_and_metrics[k] = nu.sum_multi_gpu_blob(k)
            else:
                self.losses_and_metrics[k] = nu.average_multi_gpu_blob(k)
        for k, v in self.smoothed_losses_and_metrics.items():
            v.AddValue(self.losses_and_metrics[k])
            self.writer.add_scalar(k, self.losses_and_metrics[k], i_iter)
        self.iter_total_loss = np.sum(
            np.array([self.losses_and_metrics[k] for k in self.model.losses])
        )
        self.smoothed_total_loss.AddValue(self.iter_total_loss)
        # self.smoothed_mb_qsize.AddValue(
        #     self.model.roi_data_loader._minibatch_queue.qsize()
        # )

    def LogIterStats(self, cur_iter, lr):
        """Log the tracked statistics."""
        if (cur_iter % self.LOG_PERIOD == 0 or
                cur_iter == cfg.SOLVER.MAX_ITER - 1):
            stats = self.GetStats(cur_iter, lr)
            log_json_stats(stats)

    def GetStats(self, cur_iter, lr):
        eta_seconds = self.iter_timer.average_time * (
            cfg.SOLVER.MAX_ITER - cur_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        mem_stats = c2_py_utils.GetGPUMemoryUsageStats()
        mem_usage = np.max(mem_stats['max_by_gpu'][:cfg.NUM_GPUS])
        stats = dict(
            iter=cur_iter,
            lr=float(lr),
            time=self.iter_timer.average_time,
            loss=self.smoothed_total_loss.GetMedianValue(),
            eta=eta,
            # mb_qsize=int(
            #     np.round(self.smoothed_mb_qsize.GetMedianValue())
            # ),
            mem=int(np.ceil(mem_usage / 1024 / 1024))
        )
        for k, v in self.smoothed_losses_and_metrics.items():
            stats[k] = v.GetMedianValue()
        return stats
    
    def SaveTrainingStates(self, save_file):
        save_state = {
                key:0 for key in self.smoothed_losses_and_metrics.keys()
                }
        save_smooth_state = {
                key:0 for key in self.smoothed_losses_and_metrics.keys()
                }
        for k, v in self.smoothed_losses_and_metrics.items():
            save_state[k] = v.series
            save_smooth_state[k] = v.smooth_series
        save_state['total_loss'] = self.smoothed_total_loss.series
        save_smooth_state['total_loss'] = self.smoothed_total_loss.series
#        print(save_state)
        with open(save_file, 'w') as f:
            json.dump(save_state, f)
        # save smooth state
        save_smooth_file = save_file.replace('.json', '_smooth.json')
        with open(save_smooth_file, 'w') as f:
            json.dump(save_smooth_state, f)
        
    
