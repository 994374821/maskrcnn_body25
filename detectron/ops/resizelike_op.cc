#include "resizelike_op.h"

#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

void resizeNearest2x(
    int batch_size,
    int num_channels,
    int input_height,
    int input_width,
    const float* input,
    float* output) {
  const int output_height = input_height * 2;
  const int output_width = input_width * 2;
  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int y = 0; y < output_height; ++y) {
        const int in_y = y / 2;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        int vecW = (input_width / 4) * 4; // round down
        int x = 0;
        for (; x < vecW; x += 4) {
          // load 0 1 2 3
          float32x4_t v = vld1q_f32(input + in_y * input_width + x);
          const int oidx = output_width * y + x * 2;
          float32x4x2_t v2 = {{v, v}};
          // store 00 11 22 33
          vst2q_f32(output + oidx + 0, v2);
        }

        // handle remainder
        for (; x < input_width; ++x) {
          const float v = input[in_y * input_width + x];
          const int oidx = output_width * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#else
        for (int x = 0; x < input_width; ++x) {
          const float v = input[in_y * input_width + x];
          const int oidx = output_width * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#endif
      }
      input += input_height * input_width;
      output += output_height * output_width;
    }
  }
}

template <>
bool ResizeNearestLikeOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& S = Input(1);
  auto* Y = Output(0);

  const int batch_size = X.dim32(0),
            num_channels = X.dim32(1),
            input_height = X.dim32(2),
            input_width = X.dim32(3),
            S_height = S.dim32(2),
            S_width = S.dim32(3);
  float width_scale_ = S_width * 1.0 / input_width;
  float height_scale_ = S_height * 1.0 / input_height;
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, num_channels, output_height, output_width);

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();

  // Specialized implementation for fast 2x upsampling
  if (width_scale_ == 2.0 && height_scale_ == 2.0) {
    resizeNearest2x(
        batch_size, num_channels, input_height, input_width, Xdata, Ydata);
    return true;
  }

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int y = 0; y < output_height; ++y) {
        const int in_y = std::min((int)(y / height_scale_), (input_height - 1));
        for (int x = 0; x < output_width; ++x) {
          const int in_x = std::min((int)(x / width_scale_), (input_width - 1));
          Ydata[output_width * y + x] = Xdata[input_width * in_y + in_x];
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }

  return true;
}

template <>
bool ResizeNearestLikeGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  const auto& S = Input(2);
  auto* dX = Output(0);

  const auto& inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0),
            num_channels = dY.dim32(1),
            input_height = dY.dim32(2),
            input_width = dY.dim32(3);
  const int output_height = X.dim32(2);
  const int output_width = X.dim32(3);

  const int S_height = S.dim32(2);
  const int S_width = S.dim32(3);
  float width_scale_ = S_width * 1.0 / output_width;
  float height_scale_ = S_height * 1.0 / output_height;
  dX->Resize(batch_size, num_channels, output_height, output_width);
  math::Set<float, CPUContext>(dX->size(),
                               0.0f,
                               dX->mutable_data<float>(),
                               &context_);

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int y = 0; y < input_height; ++y) {
        const int out_y = std::min((int)(y / height_scale_),
                                   (output_height - 1));
        for (int x = 0; x < input_width; ++x) {
          const int out_x = std::min((int)(x / width_scale_),
                                     (output_width - 1));
          dXdata[output_width * out_y + out_x] += dYdata[input_width * y + x];
        }
      }
      dYdata += input_height * input_width;
      dXdata += output_height * output_width;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(ResizeNearestLike, ResizeNearestLikeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ResizeNearestLikeGradient,
                      ResizeNearestLikeGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(ResizeNearestLike)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .SetDoc(R"DOC(
Resizes the spatial dimensions of the input using nearest neighbor
interpolation. The `width_scale` and `height_scale` arguments
control the size of the output, which is given by:
output_width = floor(input_width * width_scale)
output_height = floor(output_height * height_scale)
)DOC")
    .Input(0, "X", "Input tensor")
	.Input(1, "S", "Input tensor")
    .Output(0, "Y", "Output tensor");

// Input: dY, output: dX
OPERATOR_SCHEMA(ResizeNearestLikeGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension");

class ResizeNearestLikeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef("ResizeNearestLikeGradient",
                             "",
                             vector<string>{GO(0), I(0), I(1)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ResizeNearestLike, ResizeNearestLikeGradient);

} // namespace caffe2
