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

  for (int m = 0; m < M_; ++m) {  
    caffe_gpu_mul(D_, B, bottom_data+m*D_, top_data+m*D_);

    FHT(top[0]->mutable_cpu_data()+m*D_, D_);

    permutateMatrix(top[0]->mutable_cpu_data()+m*D_, PI, D_, false);
    // store for backward computation
    caffe_copy(D_, top[0]->mutable_cpu_data()+m*D_, PIHBh.mutable_cpu_data()+m*D_);

    top[0]->mutable_gpu_data();
    caffe_gpu_mul(D_, G, top_data+m*D_, top_data+m*D_);
   
    FHT(top[0]->mutable_cpu_data()+m*D_, D_);
    // store for backward computation
    caffe_copy(D_, top[0]->mutable_cpu_data()+m*D_, HGPIHBh.mutable_cpu_data()+m*D_); 

    top[0]->mutable_gpu_data();
    caffe_gpu_mul(D_, S, top_data+m*D_, top_data+m*D_);
  }

  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[4]->gpu_data(), (Dtype)1., top[0]->mutable_gpu_data());
  }
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

  Blob<Dtype>* S_diff = new Blob<Dtype>(1, D_, 1, 1);
  Blob<Dtype>* G_diff = new Blob<Dtype>(1, D_, 1, 1);
  Blob<Dtype>* B_diff = new Blob<Dtype>(1, D_, 1, 1);

  caffe_gpu_set(D_, Dtype(0.0), this->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_set(D_, Dtype(0.0), this->blobs_[1]->mutable_gpu_diff());
  caffe_gpu_set(D_, Dtype(0.0), this->blobs_[2]->mutable_gpu_diff());

  for (int m = 0; m < M_; ++m) {  
    // compute S_diff
    caffe_gpu_mul(D_, top_diff+m*D_, HGPIHBh.gpu_data()+m*D_, S_diff->mutable_gpu_diff());

    // compute G_diff
    caffe_gpu_mul(D_, top_diff+m*D_, S, G_diff->mutable_gpu_diff());
    FHT(G_diff->mutable_cpu_diff(), D_);
    caffe_gpu_mul(D_, G_diff->mutable_gpu_diff(), PIHBh.gpu_data()+m*D_, G_diff->mutable_gpu_diff());

    // compute bottom_diff&&B_diff
    caffe_gpu_mul(D_, top_diff+m*D_, S, B_diff->mutable_gpu_diff());

    FHT(B_diff->mutable_cpu_diff(), D_);
    caffe_gpu_mul(D_, B_diff->mutable_gpu_diff(), G, B_diff->mutable_gpu_diff());
    permutateMatrix(B_diff->mutable_cpu_diff(), PI, D_, true);

    FHT(B_diff->mutable_cpu_diff(), D_);
     
    caffe_gpu_mul(D_, B_diff->mutable_gpu_diff(), B, bottom_diff+m*D_);
    caffe_gpu_mul(D_, B_diff->mutable_gpu_diff(), bottom_data+m*D_, B_diff->mutable_gpu_diff());

    caffe_gpu_axpy(D_, Dtype(1.0), S_diff->mutable_gpu_diff(), this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_axpy(D_, Dtype(1.0), G_diff->mutable_gpu_diff(), this->blobs_[1]->mutable_gpu_diff());
    caffe_gpu_axpy(D_, Dtype(1.0), B_diff->mutable_gpu_diff(), this->blobs_[2]->mutable_gpu_diff());
  }

  if (bias_term_) {
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top[0]->gpu_diff(),
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[4]->mutable_gpu_diff());
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(FastFoodLayer);
}  // namespace caffe
