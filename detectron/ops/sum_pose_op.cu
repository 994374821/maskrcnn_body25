
#include "sum_pose_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2{
	
	
namespace{
	
__global__ void SumPoseKernel(const int N, const int D,
    const int H, const int W, const int Cout, const float* Xdata, float* Y_data) {
  CUDA_1D_KERNEL_LOOP(index, N * H * W) {
    int x = index % W;
    int y = (index / W) % H;
    int i = index / W / H ;

    float sum = 0;
	for (int c=0; c<D; c++){
		sum += Xdata[i*(W*H*D)+c*H*W+y*W+x];
	}
	for (int c=0; c<Cout; c++){
		Y_data[i*(W*H*Cout)+c*H*W+y*W+x] = sum;
	}
	
  }
}
	
	
}

template <>
bool SumPoseOp<float, CUDAContext>::RunOnDevice() {
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
  math::Set<float, CUDAContext>(
      Y->size(), 0.f, Y->mutable_data<float>(), &context_);

  // Set output elements at even indices to zero.
  auto* Y_data = Y->mutable_data<float>();
  const float* Xdata = X.data<float>();

  SumPoseKernel<<<CAFFE_GET_BLOCKS(N * H * W),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>
		   (N, D, H, W, Cout_, Xdata, Y_data);
  
  return true; 
  }

 


REGISTER_CUDA_OPERATOR(SumPose, SumPoseOp<float, CUDAContext>);
	
}