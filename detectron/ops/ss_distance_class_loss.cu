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

#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "ss_distance_class_loss.h"

namespace caffe2 {

namespace {

__global__ void SpatialSoftmaxKernel(const int N, const int A,
    const int H, const int W, const float* Xdata, float* Pdata, int* Pclass,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, N * A * H * W) {
    int D = num_classes * A;
    int x = index % W;
    int y = (index / W) % H;
    int a = (index / (W * H)) % A;
    int i = index / W / H / A;

    // Subtract max on each cell for numerical reasons
    float max_val = -FLT_MAX;
    for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
      int idx = i * (H * W * D) +  c * (H * W) + y * W + x;
      max_val = max(max_val, Xdata[idx]);
	  if (Xdata[idx]==max_val){
		  Pclass[index] = c- a * num_classes; // the anchor predict to which class(0-20)
	  }
    }
    // Exponentiate sigmoid g(z)=1/(1+exp(-z))
    for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
      int idx = i * (H * W * D) + c * (H * W) + y * W + x;
      float expx = 1./ (1 + exp(0-Xdata[idx]));
      Pdata[idx] = expx;
    }
  }
}

// apply ss_distance weight to sigmoid entropy loss
/*__global__ void SoftmaxFocalLossKernel(
    const int N, const int A, const int H, const int W,
    const float* Pdata, const int* targets, const int* Pclass, float* losses,
    const float* gt_centers, const float* rois, const float* bbox_pred, const float* im_info, 
    const int num_classes, float* regression_w) {
  CUDA_1D_KERNEL_LOOP(i, N * A * H * W) {
    int D = A * num_classes;
    int x = i % W;
    int y = (i / W) % H;
    int a = (i / (W * H)) % A;
    int n = i / (W * H * A);
    const int label = static_cast<int>(targets[i]);
    const int Plabel = static_cast<int>(Pclass[i]);

    losses[i] = 0.0;
    if (label >= 0 and(Plabel==14 or Plabel==15 or Plabel==16 or Plabel==17 or Plabel==18 or Plabel==19) ){
      int offset = a * num_classes;
      int idx = n * (H * W * D) + (offset + Plabel) * (H * W) + y * W + x;
	  if(Plabel==label){
		  losses[i] =
          -log(max(Pdata[idx], FLT_MIN)) ;
	  }
	  else{
          int indx_img=rois[n * (H * W * D) + (a * 5 + 0) * (H * W) + y * W + x] ;
        
          float im_scale = im_info[indx_img* 3+2];
          float im_h = im_info[indx_img * 3 + 0];
          float im_w = im_info[indx_img * 3 + 1];
          float wx = regression_w[indx_img * 4 + 0];
          float wy = regression_w[indx_img * 4 + 1];
          float x0 = rois[n * (H * W * D) + (a * 5 + 1) * (H * W) + y * W + x] / im_scale;
          float y0 = rois[n * (H * W * D) + (a * 5 + 2) * (H * W) + y * W + x] / im_scale;
          float x1 = rois[n * (H * W * D) + (a * 5 + 3) * (H * W) + y * W + x] / im_scale;
          float y1 = rois[n * (H * W * D) + (a * 5 + 4) * (H * W) + y * W + x] / im_scale;
          float x_ca = x0 + (x1-x0)/2;
          float y_ca = y0 + (y1-y0)/2;
          float wa = x1 - x0;
          float ha = y1 - y0;
          float tx = bbox_pred[n * (H * W * D) + (a * num_classes * 4 + Plabel * 4 + 0) * (H * W) + y * W + x];
          float ty = bbox_pred[n * (H * W * D) + (a * num_classes * 4 + Plabel * 4 + 1) * (H * W) + y * W + x];
          float x_c = tx / wx * wa + x_ca; // pred x center
          float y_c  = ty / wy * ha + y_ca; // pred y center
          
          float gt_xc = gt_centers[12*indx_img + (Plabel-14)*2+0];
          float gt_yc = gt_centers[12*indx_img + (Plabel-14)*2+1];
          float dis = 1;
          if(gt_xc!=-1){ // this gt boxes not exist
              dis = sqrt((x_c-gt_xc) * (x_c-gt_xc) + (y_c - gt_yc) * (y_c - gt_yc)) / (sqrt(im_h*im_h + im_w*im_w));
          }
		  losses[i] = -log(1-max(Pdata[idx], FLT_MIN))* dis;
	  }
    }
  }
}*/

__global__ void SoftmaxFocalLossKernel(
    const int N, const int A, const int H, const int W,
    const float* Pdata, const int* targets, const int* Pclass, float* losses,
    const float* gt_centers, const float* rois, const float* im_info, const float* bbox_pred,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * A * H * W) {
    int D = A * num_classes;
    int x = i % W;
    int y = (i / W) % H;
    int a = (i / (W * H)) % A;
    int n = i / (W * H * A);
    const int label = static_cast<int>(targets[i]);
    const int Plabel = static_cast<int>(Pclass[i]);

    losses[i] = 0.0;
    if (label >= 0 and(Plabel==14 or Plabel==15 or Plabel==16 or Plabel==17 or Plabel==18 or Plabel==19) ){
      int offset = a * num_classes;
      int idx = n * (H * W * D) + (offset + Plabel) * (H * W) + y * W + x;
	  if(Plabel==label){
		  losses[i] =
          -log(max(Pdata[idx], FLT_MIN)) ;
	  }
	  else{
          int indx_img=rois[n * (H * W * D) + (a * 5 + 0) * (H * W) + y * W + x] ;
        
          float im_scale = im_info[indx_img* 3+2];
          float im_h = im_info[indx_img * 3 + 0];
          float im_w = im_info[indx_img * 3 + 1];
          // float wx = regression_w[indx_img * 4 + 0];
          // float wy = regression_w[indx_img * 4 + 1];
          float x0 = rois[n * (H * W * D) + (a * 5 + 1) * (H * W) + y * W + x] / im_scale;
          float y0 = rois[n * (H * W * D) + (a * 5 + 2) * (H * W) + y * W + x] / im_scale;
          float x1 = rois[n * (H * W * D) + (a * 5 + 3) * (H * W) + y * W + x] / im_scale;
          float y1 = rois[n * (H * W * D) + (a * 5 + 4) * (H * W) + y * W + x] / im_scale;
          float x_ca = x0 + (x1-x0)/2;
          float y_ca = y0 + (y1-y0)/2;
          float wa = x1 - x0;
          float ha = y1 - y0;
          float tx = bbox_pred[n * (H * W * D) + (a * num_classes * 4 + Plabel * 4 + 0) * (H * W) + y * W + x];
          float ty = bbox_pred[n * (H * W * D) + (a * num_classes * 4 + Plabel * 4 + 1) * (H * W) + y * W + x];
          float wx = 1;
          float wy = 1;
          float x_c = tx / wx * wa + x_ca; // pred x center
          float y_c  = ty / wy * ha + y_ca; // pred y center
          //float x_c  = x_ca;
          //float y_c = y_ca;
          
          float gt_xc = gt_centers[12*indx_img + (Plabel-14)*2+0];
          float gt_yc = gt_centers[12*indx_img + (Plabel-14)*2+1];
          float dis = 1;
          if(gt_xc!=-1){ // this gt boxes exist
              dis = sqrt((x_c-gt_xc) * (x_c-gt_xc) + (y_c - gt_yc) * (y_c - gt_yc)) / (sqrt(im_h*im_h + im_w*im_w));
          }
		  losses[i] = -log(1-max(Pdata[idx], FLT_MIN))* dis;
	  }
    }
  }
}


__global__ void SoftmaxFocalLossGradientKernel(
    const int N, const int D, const int H, const int W,
    const float* Pdata, const int* targets, const int* Pclass, const float* d_loss_data, float* dX, 
    const float* gt_centers, const float* rois, const float* im_info, const float* bbox_pred, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * D * H * W) {
    int A = D / num_classes;
    int x = i % W;
    int y = (i / W) % H;
    int d = (i / (W * H)) % D;
    int a = d / num_classes;  // this is a th anchor
    int c = d % num_classes; // this is c th class index
    int n = i / (W * H * D);
    float d_loss = *d_loss_data;

    int ind = n * (H * W * A) + a * (H * W) + y * W + x;
    const int label = static_cast<int>(targets[ind]);
    const int Plabel = static_cast<int>(Pclass[ind]);
    
    dX[i] = 0.0;
    if (label >= 0  and c==Plabel and(Plabel==14 or Plabel==15 or Plabel==16 or Plabel==17 or Plabel==18 or Plabel==19) ){
        //printf("c==Plabel, c:%d, Plabel:%d\n", c, Plabel);
        if (Plabel==label){
            dX[i] = d_loss*(Pdata[i] - 1);
            //printf("Plabel==label dx: %f, index: %d, c: %d, Plabel:%d, label:%d\n", dX[i], i, c, Plabel, label);
        }
        else{
          int indx_img=rois[n * (H * W * D) + (a * 5 + 0) * (H * W) + y * W + x] ;
        
          float im_scale = im_info[indx_img* 3+2];
          float im_h = im_info[indx_img * 3 + 0];
          float im_w = im_info[indx_img * 3 + 1];
          // float wx = regression_w[indx_img * 4 + 0];
          // float wy = regression_w[indx_img * 4 + 1];
          float x0 = rois[n * (H * W * D) + (a * 5 + 1) * (H * W) + y * W + x] / im_scale;
          float y0 = rois[n * (H * W * D) + (a * 5 + 2) * (H * W) + y * W + x] / im_scale;
          float x1 = rois[n * (H * W * D) + (a * 5 + 3) * (H * W) + y * W + x] / im_scale;
          float y1 = rois[n * (H * W * D) + (a * 5 + 4) * (H * W) + y * W + x] / im_scale;
          float x_ca = x0 + (x1-x0)/2;
          float y_ca = y0 + (y1-y0)/2;
          float wa = x1 - x0;
          float ha = y1 - y0;
          float tx = bbox_pred[n * (H * W * D) + (a * num_classes * 4 + Plabel * 4 + 0) * (H * W) + y * W + x];
          float ty = bbox_pred[n * (H * W * D) + (a * num_classes * 4 + Plabel * 4 + 1) * (H * W) + y * W + x];
          float wx = 1;
          float wy = 1;
          float x_c = tx / wx * wa + x_ca; // pred x center
          float y_c  = ty / wy * ha + y_ca; // pred y center
          //float x_c  = x_ca;
          //float y_c = y_ca;
          
          float gt_xc = gt_centers[12*indx_img + (Plabel-14)*2+0];
          float gt_yc = gt_centers[12*indx_img + (Plabel-14)*2+1];
          float dis = 1;
          if(gt_xc!=-1){ // this gt boxes exist
              dis = sqrt((x_c-gt_xc) * (x_c-gt_xc) + (y_c - gt_yc) * (y_c - gt_yc)) / (sqrt(im_h*im_h + im_w*im_w));
          }
		  dX[i] = d_loss*Pdata[i]* dis;
        }
    }

  }
}

} // namespace


template <>
bool SsDistanceLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);         // Logits
  auto& T = Input(1);         // Labels
  auto& rois = Input(2);        // 
  auto& bbox_pred = Input(3);         // Logits
  auto& gt_centers = Input(4);         // Labels
  auto& im_info = Input(5);        // num of foregound normalizer
  auto* avg_loss = Output(0); // average loss as output
  auto* P = Output(1);        // softmax probability, going to be re-used in gradient
  auto* Plabel = Output(2);     //predict classes of each anchor(max score)

  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int A = D / num_classes_;

  losses_.Resize(N * A * H * W);
  P->Resize(N * D * H * W);
  Plabel->Resize(N * A * H * W);
  avg_loss->Resize(vector<TIndex>());
  math::Set<float, CUDAContext>(
      avg_loss->size(), 0.f, avg_loss->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
      P->size(), 0.f, P->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
      losses_.size(), 0.f, losses_.mutable_data<float>(), &context_);
  math::Set<int, CUDAContext>(
      Plabel->size(), 0.f, Plabel->mutable_data<int>(), &context_); // predict class
  DCHECK_EQ(X.ndim(), 4);

  const float* Xdata = X.data<float>();

  // Spatial Softmax Kernel
  SpatialSoftmaxKernel
      <<<CAFFE_GET_BLOCKS(N * A * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, A, H, W, Xdata, P->mutable_data<float>(), Plabel->mutable_data<int>(), num_classes_);

  // Compute loss for each x,y location
  const int* Tdata = T.data<int>();
  const float* RoiData = rois.data<float>();
  const float* bbox_pred_data = bbox_pred.data<float>();
  const float* gt_centers_data = gt_centers.data<float>();
  const float* im_info_data = im_info.data<float>();
/*   SoftmaxFocalLossKernel
  <<<CAFFE_GET_BLOCKS(N * A * H * W), CAFFE_CUDA_NUM_THREADS,
      0, context_.cuda_stream()>>>(
    N, A, H, W, P->data<float>(), Tdata, Plabel->data<int>(), losses_.mutable_data<float>(),
    gt_centers_data, RoiData, bbox_pred_data, im_info_data, num_classes_, &regression_ws_[0]); */
    
  SoftmaxFocalLossKernel
  <<<CAFFE_GET_BLOCKS(N * A * H * W), CAFFE_CUDA_NUM_THREADS,
      0, context_.cuda_stream()>>>(
    N, A, H, W, P->data<float>(), Tdata, Plabel->data<int>(), losses_.mutable_data<float>(),
    gt_centers_data, RoiData, im_info_data, bbox_pred_data, num_classes_);

  // sum the losses
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  math::Scale<float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);

  return true;
}

template<>
bool SsDistanceLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);    // Logits
  auto& T = Input(1);    // Label
  auto& rois = Input(2);        // [im_batch, x0, y0, x1, y1]
  auto& bbox_pred = Input(3);         
  auto& gt_centers = Input(4);         
  auto& im_info = Input(5);        // [w, h, im_scale]
  auto& P = Input(6);    // sigmoid Probability
  auto& Plabel = Input(7); // predict anchors belong to which class 
  auto& d_avg_loss = Input(8);
  auto* dX = Output(0);  // gradient wrt logits


  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int A = D / num_classes_;

  dX->ResizeLike(X);

  const float* Xdata = X.data<float>();
  const int* Tdata = T.data<int>();
  const float* Pdata = P.data<float>();
  const float* RoiData = rois.data<float>();
  const float* bbox_pred_data = bbox_pred.data<float>();
  const float* gt_centers_data = gt_centers.data<float>();
  const float* im_info_data = im_info.data<float>();
  const int* Plabel_data = Plabel.data<int>();
  

  SoftmaxFocalLossGradientKernel
      <<<CAFFE_GET_BLOCKS(N * D * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, D, H, W, Pdata, Tdata, Plabel_data, d_avg_loss.data<float>(), dX->mutable_data<float>(), 
    gt_centers_data, RoiData, im_info_data, bbox_pred_data, num_classes_);
  math::Scale<float, CUDAContext>(
    dX->size(), scale_, dX->data<float>(), dX->mutable_data<float>(),
    &context_);
  return true;
}


REGISTER_CUDA_OPERATOR(SsDistanceLoss,
                       SsDistanceLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SsDistanceLossGradient,
                       SsDistanceLossGradientOp<float, CUDAContext>);

} // namespace caffe2
