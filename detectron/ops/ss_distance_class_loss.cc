/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ss_distance_class_loss.h"
#include "caffe2/operators/softmax_shared.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SsDistanceLoss, SsDistanceLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SsDistanceLossGradient,
    SsDistanceLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SsDistanceLoss)
    .NumInputs(6)
    .NumOutputs(3)
    .SetDoc(R"DOC(
A multiclass form of Focal Loss designed for use in RetinaNet-like models.
The input is assumed to be unnormalized scores (sometimes called 'logits')
arranged in a 4D tensor with shape (N, C, H, W), where N is the number of
elements in the batch, H and W are the height and width, and C = num_anchors *
num_classes. The softmax is applied num_anchors times along the C axis.

The softmax version of focal loss is:

  FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t),

where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
s_j is the unnormalized score for class j.

See: https://arxiv.org/abs/1708.02002 for details.
)DOC")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Arg(
        "num_classes",
        "(int) default 81; number of classes in each softmax group.")
    .Input(
        0,
        "scores",
        "4D tensor of softmax inputs (called 'scores' or 'logits') with shape "
        "(N, C, H, W), where C = num_anchors * num_classes defines num_anchors "
        "groups of contiguous num_classes softmax inputs.")
    .Input(
        1,
        "labels",
        "4D tensor of labels with shape (N, num_anchors, H, W). Each entry is "
        "a class label in [0, num_classes - 1] (inclusive).")
    .Input(
        2,
        "rois",
        "rois; proposed rois. (batch_idx, x1, y1, x2, y2)"
        "4D tensor whith shape (1, 5*num_anchors, 1, 1)"
    )
	.Input(
	    3,
	    "bbox_pred (4*num_classes)",
		"bbox regression with shape (1, num_anchors*4*num_classes, 1, 1)"
	)
	.Input(
	    4,
	    "gt_centers",
		"gt_centers shape is (im_batch, 6*2); x center, y center;"
        "if that gt_box is none, the centre is -1"
    )
    .Input(
	    5,
	    "im_info",
		"gt_centers shape is (im_batch, 6); if that gt_box is none, the centre is -1"
	)
    .Output(
        0,
        "loss",
        "Scalar loss.")
    .Output(
        1,
        "probabilities",
        "4D tensor of softmax probabilities with shape (N, C, H, W), where "
        "C = num_anchors * num_classes, and softmax was applied to each of the "
        "num_anchors groups; within a group the num_classes values sum to 1.")
      .Output(
        2,
        "pred_classes",
        "4D tensor of softmax probabilities with shape (N, A, H, W), where "
        "A is num_anchors ");

OPERATOR_SCHEMA(SsDistanceLossGradient)
    .NumInputs(9)
    .NumOutputs(1)
    .Input(
        0,
        "scores",
        "See SoftmaxFocalLoss.")
    .Input(
        1,
        "labels",
        "See SoftmaxFocalLoss.")
    .Input(
        2,
        "rois",
        "See SoftmaxFocalLoss.")
    .Input(
        3,
        "bbox_pred",
        "See SoftmaxFocalLoss..")
    .Input(
	    4,
	    "gt_centers",
		"gt_centers shape is (im_batch, 6*2); x center, y center;"
        "if that gt_box is none, the centre is -1"
    )
    .Input(
	    5,
	    "im_info",
		"gt_centers shape is (im_batch, 6); if that gt_box is none, the centre is -1"
	)
    .Input(
	    6,
	    "probabilities",
		"output 1 of SsDistanceLossop"
	)
    .Input(
	    7,
	    "pred_classes",
		"output 2 of SsDistanceLossop"
	)
    .Input(
        8,
        "d_loss",
        "Gradient of forward output 0 (loss)")
    .Output(
        0,
        "d_scores",
        "Gradient of forward input 0 (scores)");

class GetSsDistanceLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SsDistanceLossGradient",
        "",
        vector<string>{I(0), I(1), I(2), I(3), I(4), I(5), O(1), O(2), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SsDistanceLoss, GetSsDistanceLossGradient);

} // namespace caffe2
