#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FastFoodLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  // input dim
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.fastfood_param().axis());
      //this->layer_param_.inner_product_param().axis());
  D_ = bottom[0]->count(axis);
  top[0]->Reshape(bottom[0]->num(), D_, 1, 1); // It's ok if D_ is beyong the range of labels
  // output dim
  N_ = this->layer_param_.fastfood_param().num_output();

  CHECK_EQ(D_, N_) << "Input dim must agree with the output dim";

  CHECK_EQ(checkDimension(D_), true) << "Input dim must be a power of two"; 
  PIHBh = new Dtype[D_];
  HGPIHBh = new Dtype[D_];

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    srand(time(NULL));
    int _param_num = 4;
    this->blobs_.resize(_param_num);
    // Intialize the weight
    for (int i = 0; i < _param_num; ++i) {
      vector<int> weight_shape(1, D_);
      this->blobs_[i].reset(new Blob<Dtype>(weight_shape));
      if (i == 0) {
        shared_ptr<Filler<Dtype> > s_filler(GetFiller<Dtype>(
            this->layer_param_.fastfood_param().s_filler()));
        s_filler->Fill(this->blobs_[i].get());
      } else if (i == 1) {
        shared_ptr<Filler<Dtype> > g_filler(GetFiller<Dtype>(
            this->layer_param_.fastfood_param().g_filler()));
        g_filler->Fill(this->blobs_[i].get());
      } else if (i == 2) {
        shared_ptr<Filler<Dtype> > b_filler(GetFiller<Dtype>(
            this->layer_param_.fastfood_param().b_filler()));
        b_filler->Fill(this->blobs_[i].get());
      }
    }

    // \pi is determined when layer is setup (the same in test phase)
    vector<Dtype> permutation_idx; 
    for (int i = 0; i < D_; ++i)
      permutation_idx.push_back((Dtype)i);
    random_shuffle(permutation_idx.begin(), permutation_idx.end());
    Dtype* buffer = this->blobs_[3]->mutable_cpu_data();
    for (int j = 0 ; j < D_; ++j){
      buffer[j] = permutation_idx[j];
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void FastFoodLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); // h_l
  Dtype* top_data = top[0]->mutable_cpu_data(); // h_{l+1}

  const Dtype* S = this->blobs_[0]->cpu_data();
  const Dtype* G = this->blobs_[1]->cpu_data();
  const Dtype* B = this->blobs_[2]->cpu_data();
  const Dtype* PI = this->blobs_[3]->cpu_data();

  caffe_mul(D_, B, bottom_data, top_data);
  FHT(top_data, D_);
  permutateMatrix(top_data, PI, D_, false);
  // store for backward computation
  caffe_copy(D_, top_data, PIHBh);
  caffe_mul(D_, G, top_data, top_data);
  FHT(top_data, D_);
  // store for backward computation
  caffe_copy(D_, top_data, HGPIHBh);
  caffe_mul(D_, S, top_data, top_data);
}


template <typename Dtype>
void FastFoodLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Dimension of h_{l}, h_{l+1}_diff is 'd*1', but they are stored as '1*d' array.
  // The dialog elements of '(d*1)*(1*d)' are the same as element-wise multiplication.
  const Dtype* top_diff = top[0]->cpu_diff(); // d(E)/d(h_{l+1})
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* S = this->blobs_[0]->cpu_data();
  const Dtype* G = this->blobs_[1]->cpu_data();
  const Dtype* B = this->blobs_[2]->cpu_data();
  const Dtype* PI = this->blobs_[3]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* S_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* G_diff = this->blobs_[1]->mutable_cpu_diff();
  Dtype* B_diff = this->blobs_[2]->mutable_cpu_diff();

  // compute S_diff
  caffe_mul(D_, top_diff, HGPIHBh, S_diff);

  // compute G_diff
  caffe_mul(D_, top_diff, S, G_diff);
  FHT(G_diff, D_);
  caffe_mul(D_, G_diff, PIHBh, G_diff);

  // compute bottom_diff&&B_diff
  caffe_mul(D_, top_diff, S, B_diff);
  FHT(B_diff, D_);
  caffe_mul(D_, B_diff, G, B_diff);
  permutateMatrix(B_diff, PI, D_, true);
  FHT(B_diff, D_);

  caffe_mul(D_, B_diff, B, bottom_diff);
  caffe_mul(D_, B_diff, bottom_data, B_diff);

  // for (int i = 0 ; i < D_; ++i){
  //    this->blobs_[3]->mutable_cpu_data()[i] = ceil(this->blobs_[3]->cpu_data()[i]);
  // }
}

template <typename Dtype>
bool FastFoodLayer<Dtype>::checkDimension(size_t length)
{
  if (length <= 0) return false;
  if (length&(length-1))
    return false;
  else
    return true;
}

template <typename Dtype>
void FastFoodLayer<Dtype>::permutateMatrix(Dtype* data, const Dtype* permutation, size_t length, bool trans) {
  Dtype* buffer = new Dtype[length];
  if (!trans) {   
    for (size_t i = 0 ; i < length; ++i) {
      buffer[i] = data[(int)(permutation[i]+0.5)];
    }
  } else {
    for (size_t i = 0; i < length; ++i) {
      buffer[(int)(permutation[i]+0.5)] = data[i];
    }
  }
  for (size_t i = 0 ; i < length; ++i) {
    data[i] = buffer[i];
  }  
}

/**
 * Fast Hadamard Transform
 *
 * 'data(input vector)' is modifed in-place to become the output vector.
 * 'length(#elements of data)' should be a power of 2.
 *
 * TODO: Adding normalisation factors?
 */
template <typename Dtype>
void FastFoodLayer<Dtype>::FHT(Dtype* data, size_t length)
{
  Dtype lvalue, rvalue = 0.0;

  for (size_t size = length / 2; size; size /= 2) {
    for(size_t step = 0; step < length; step += (size*2)) {
      for(size_t index = 0; index < size; index++) {

        lvalue = data[step + index];
        rvalue = data[step + index + size];
        
        data[step + index]        = lvalue + rvalue;
        data[step + index + size] = lvalue - rvalue;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FastFoodLayer);
#endif

INSTANTIATE_CLASS(FastFoodLayer);
REGISTER_LAYER_CLASS(FastFood);
}  // namespace caffe
