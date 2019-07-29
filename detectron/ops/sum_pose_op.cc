
#include "sum_pose_op.h"

namespace caffe2{
	
template <>
bool SumPoseOp<float, CPUContext>::RunOnDevice() {
  // Retrieve the input tensor.
  const auto& X = Input(0);
  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  // Initialize the output tensor to a copy of the input tensor.
  auto* Y = Output(0);
  //Y->Resize(N * Cout_ * H * W);
  vector<int> outShape={N, Cout_, H, W};
  Y->Resize(outShape);
  
  //printf("");
  math::Set<float, CPUContext>(
      Y->size(), 0.f, Y->mutable_data<float>(), &context_);

  // Set output elements at even indices to zero.
  auto* Y_data = Y->mutable_data<float>();
  const float* Xdata = X.data<float>();
  for (auto index = 0; index < N*H*W; index ++) {
    int x = index % W;
	int y = (index / W) % H;
	int i = index / W / H ;
	
	float sum = 0;
	for (int c=0; c<D; c++){
		sum += Xdata[i*(W*H*D)+c*H*W+y*W+x];
	}
	for (int c=0; c<Cout_; c++){
		Y_data[i*(W*H*Cout_)+c*H*W+y*W+x] = sum;
	}
	
  }

  return true;
}

REGISTER_CPU_OPERATOR(SumPose, SumPoseOp<float, CPUContext>);

OPERATOR_SCHEMA(SumPose)
    .NumInputs(1)
    .NumOutputs(1)
	.Arg(
        "Cout",
        "(float) default 1.0; the channel of output  Y (channel out).")
    .Input(
        0,
        "X",
        "1D input tensor")
    .Output(
        0,
        "Y",
        "1D output tensor");	
	
	
	
	
}