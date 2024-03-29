#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "resizelike_op.h"

namespace caffe2 {

namespace {

__global__ void NearestNeighborKernel(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(index, size) {

    int indexTemp = index;
    const int w = indexTemp % output_width;
    indexTemp /= output_width;
    const int h = indexTemp % output_height;
    indexTemp /= output_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int in_y = fminf(h / height_scale, input_height - 1);
    const int in_x = fminf(w / width_scale, input_width - 1);
    Y[index] =
        X[((n * num_channels + c) * input_height + in_y) * input_width + in_x];
  }
}

__global__ void NearestNeighborGradientKernel(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* dY,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int x = indexTemp % input_width;
    indexTemp /= input_width;
    const int y = indexTemp % input_height;
    indexTemp /= input_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int out_y = fminf(y / height_scale, output_height - 1);
    const int out_x = fminf(x / width_scale, output_width - 1);
    const int out_index =
        ((n * num_channels + c) * output_height + out_y) * output_width + out_x;
#if __CUDA_ARCH__ >= 350
    atomicAdd(dX + out_index, __ldg(dY + index));
#else
    atomicAdd(dX + out_index, *(dY + index));
#endif
  }
}

} // namespace

template <>
bool ResizeNearestLikeOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  const auto& S = Input(1);

  const auto& inputDims = X.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_height = X.dim32(2), input_width = X.dim32(3),
            S_height = S.dim32(2),
            S_width = S.dim32(3);
  float width_scale_ = S_width * 1.0 / input_width;
  float height_scale_ = S_height * 1.0 / input_height;
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, num_channels, output_height, output_width);

  const auto size = Y->size();
  NearestNeighborKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->mutable_data<float>());

  return true;
}

template <>
bool ResizeNearestLikeGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  const auto& S = Input(2);
  auto* dX = Output(0);

  const int S_height = S.dim32(2);
  const int S_width = S.dim32(3);

  const auto& inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
            input_height = dY.dim32(2), input_width = dY.dim32(3);
  int output_height = X.dim32(2);
  int output_width = X.dim32(3);
	
  float width_scale_ = S_width * 1.0 / output_width;
  float height_scale_ = S_height * 1.0 / output_height;

  dX->Resize(batch_size, num_channels, output_height, output_width);
  math::Set<float, CUDAContext>(
      dX->size(), 0.0f, dX->mutable_data<float>(), &context_);

  const auto size = dY.size();
  NearestNeighborGradientKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      dY.data<float>(),
      dX->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(ResizeNearestLike, ResizeNearestLikeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ResizeNearestLikeGradient,
    ResizeNearestLikeGradientOp<float, CUDAContext>);
} // namespace caffe2
