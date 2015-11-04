#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FastFoodLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data(); // h_l
  Dtype* top_data = top[0]->mutable_gpu_data(); // h_{l+1}

  const Dtype* S = this->blobs_[0]->gpu_data();
  const Dtype* G = this->blobs_[1]->gpu_data();
  const Dtype* B = this->blobs_[2]->gpu_data();
  const Dtype* PI = this->blobs_[3]->cpu_data();

  caffe_gpu_mul(D_, B, bottom_data, top_data);

  FHT(top[0]->mutable_cpu_data(), D_);

  permutateMatrix(top[0]->mutable_cpu_data(), PI, D_, false);
  // store for backward computation
  caffe_copy(D_, top[0]->mutable_cpu_data(), PIHBh);

  top[0]->mutable_gpu_data();
  caffe_gpu_mul(D_, G, top_data, top_data);
 
  FHT(top[0]->mutable_cpu_data(), D_);
  // store for backward computation
  caffe_copy(D_, top[0]->mutable_cpu_data(), HGPIHBh); 

  top[0]->mutable_gpu_data();
  caffe_gpu_mul(D_, S, top_data, top_data);

}

// TODO: GPU version of FHT
template <typename Dtype>
void FastFoodLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Dimension of h_{l}, h_{l+1}_diff is 'd*1', but they are stored as '1*d' array.
  // The dialog elements of '(d*1)*(1*d)' are the same as element-wise multiplication.
  const Dtype* top_diff = top[0]->gpu_diff(); // d(E)/d(h_{l+1})
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* S = this->blobs_[0]->gpu_data();
  const Dtype* G = this->blobs_[1]->gpu_data();
  const Dtype* B = this->blobs_[2]->gpu_data();
  const Dtype* PI = this->blobs_[3]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* S_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* G_diff = this->blobs_[1]->mutable_gpu_diff();
  Dtype* B_diff = this->blobs_[2]->mutable_gpu_diff();

  Dtype* G_diff_cpu = this->blobs_[1]->mutable_cpu_diff();
  Dtype* B_diff_cpu = this->blobs_[2]->mutable_cpu_diff();

  // compute S_diff
  Blob<Dtype>* tmp_blob = new Blob<Dtype>(1, D_, 1, 1);
  caffe_copy(D_, HGPIHBh, tmp_blob->mutable_cpu_data());
  caffe_gpu_mul(D_, top_diff, tmp_blob->gpu_data(), S_diff);


  // compute G_diff
  this->blobs_[1]->mutable_gpu_diff();
  caffe_gpu_mul(D_, top_diff, S, G_diff);

  this->blobs_[1]->mutable_cpu_diff();
  FHT(G_diff_cpu, D_);

  caffe_copy(D_, PIHBh, tmp_blob->mutable_cpu_data());
  this->blobs_[1]->mutable_gpu_diff();
  caffe_gpu_mul(D_, G_diff, tmp_blob->gpu_data(), G_diff);

  // compute bottom_diff&&B_diff
  this->blobs_[2]->mutable_gpu_diff();
  caffe_gpu_mul(D_, top_diff, S, B_diff);

  this->blobs_[2]->mutable_cpu_diff();
  FHT(B_diff_cpu, D_);

  this->blobs_[2]->mutable_gpu_diff();
  caffe_gpu_mul(D_, B_diff, G, B_diff);

  this->blobs_[2]->mutable_cpu_diff(); 
  permutateMatrix(B_diff_cpu, PI, D_, true);

  FHT(B_diff_cpu, D_);

  this->blobs_[2]->mutable_gpu_diff(); 
  caffe_gpu_mul(D_, B_diff, B, bottom_diff);
  caffe_gpu_mul(D_, B_diff, bottom_data, B_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(FastFoodLayer);
}  // namespace caffe
