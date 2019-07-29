#ifndef SS_DISTANCE_CLASS_LOSS_H_
#define SS_DISTANCE_CLASS_LOSS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SsDistanceLossOp final : public Operator<Context> {
 public:
  SsDistanceLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", 20)),
        //regression_ws_(OperatorBase::GetRepeatedArgument<float>("regression_ws")),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  //vector<float> regression_ws_;
  StorageOrder order_;
  Tensor<Context> losses_;
};

template <typename T, class Context>
class SsDistanceLossGradientOp final : public Operator<Context> {
 public:
  SsDistanceLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", 20)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  StorageOrder order_;


};

} // namespace caffe2

#endif // SS_DISTANCE_CLASS_LOSS_H_