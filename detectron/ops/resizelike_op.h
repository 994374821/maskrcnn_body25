#ifndef RESIZELIKE_OP_H_
#define RESIZELIKE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ResizeNearestLikeOp final : public Operator<Context> {
 public:
  ResizeNearestLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    //CAFFE_ENFORCE_GT(width_scale_, 0);
    //CAFFE_ENFORCE_GT(height_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

// protected:
//  T width_scale_;
//  T height_scale_;
};

template <typename T, class Context>
class ResizeNearestLikeGradientOp final : public Operator<Context> {
 public:
  ResizeNearestLikeGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
   // CAFFE_ENFORCE_GT(width_scale_, 0);
   // CAFFE_ENFORCE_GT(height_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 //protected:
 // T width_scale_;
 // T height_scale_;
};

} // namespace caffe2

#endif
