#ifndef SUM_POSE_OP_H_
#define SUM_POSE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

/**
 * ZeroEven operator. Zeros elements at even indices of an 1D array.
 * Elements at odd indices are preserved.
 *
 * This toy operator is an example of a custom operator and may be a useful
 * reference for adding new custom operators to the Detectron codebase.
 */
template <typename T, class Context>
class SumPoseOp final : public Operator<Context> {
 public:
  // Introduce Operator<Context> helper members.
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  SumPoseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) ,
	  Cout_(OperatorBase::GetSingleArgument<int>("Cout", 1.)),
	  order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(Cout_ >= 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }

  bool RunOnDevice() override;
  
 protected:
  int Cout_;
  StorageOrder order_;
  
};

} // namespace caffe2

#endif // ZERO_EVEN_OP_H_