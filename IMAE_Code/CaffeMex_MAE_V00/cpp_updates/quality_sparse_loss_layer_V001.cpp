#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/layers/quality_sparse_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SparseLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // set the values for weights_ave_
  int count_ = bottom[0]->count();
  int sample_num_ = count_ / bottom[0]->num();


  LOG(INFO) << "count_: " << count_;
  LOG(INFO) << "sample_num_: " << sample_num_;

  dist_sq_.Reshape(sample_num_, 1, 1, 1);
  diff_.Reshape(bottom[0]->num(), sample_num_, 1, 1);
  weights_ave_.Reshape(bottom[0]->num(), sample_num_, 1, 1);
  caffe_set(count_, Dtype(1.0) / Dtype(bottom[0]->num()), weights_ave_.mutable_cpu_data());

  margin = Dtype(1.0) / Dtype(bottom[0]->num());


  LOG(INFO) << "margin: "<<margin;
  LOG(INFO) << "reference: "<<Dtype(1.0) / Dtype(bottom[0]->num());
  for(int i=0; i < weights_ave_.count(); i++){
    LOG(INFO) << "weights_ave_[" << i << "]: " << weights_ave_.cpu_data()[i];
  }
}
template <typename Dtype>
void SparseLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void SparseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count_ = bottom[0]->count();
  caffe_sub(
      count_,
      bottom[0]->cpu_data(),
      weights_ave_.cpu_data(),
      diff_.mutable_cpu_data());

  for(int i=0; i < count_; i++){
    LOG(INFO) << "bottom[0][" << i << "]: " << bottom[0]->cpu_data()[i];
  }
  for(int i=0; i < count_; i++){
    LOG(INFO) << "diff_[" << i << "]: " << diff_.cpu_data()[i];
  }

  int sample_num_ = count_ / bottom[0]->num();
  loss = Dtype(0);
  for( int sn = 0; sn < sample_num_; sn++)
  {
    dist_sq_.mutable_cpu_data()[sn] = caffe_cpu_strided_dot(bottom[0]->num(), diff_.cpu_data() + sn, sample_num_, diff_.cpu_data() + sn, sample_num_);
    
    LOG(INFO) << "dist_sq_[" << sn << "]: " << dist_sq_.cpu_data()[sn];

    loss += std::max(margin - dist_sq_.cpu_data()[sn], Dtype(0.0));
  }  
  loss = loss / static_cast<Dtype>(sample_num_) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  LOG(INFO) << "sparese_loss: " << loss;
}

template <typename Dtype>
void SparseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  int count_ = bottom[0]->count();
  int sample_num_ = count_ / bottom[0]->num();
  const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(sample_num_);

  // Gradient cut: in case that the gradient is 0 or too large
  for(int i = 0; i < count_; i++){
    //no gap
    if( diff_.cpu_data()[i] < Dtype(1e-9) ){
      diff_.mutable_cpu_data()[i] = Dtype(1e-5);
    }
    //very huge gap
    if( diff_.cpu_data()[i] > Dtype(bottom[0]->num() / 2 -1) * margin ){
      diff_.mutable_cpu_data()[i] = 0;
    }
    //large gap
    if( diff_.cpu_data()[i] > Dtype(2) * margin ){
      diff_.mutable_cpu_data()[i] = margin;
    } 
    
  }

  for( int sn = 0; sn < sample_num_; sn++)
  {
    Dtype mdist = margin - dist_sq_.cpu_data()[sn];
    Dtype beta = -alpha;

    LOG(INFO) << "mdist[" << sn << "]: " << mdist;
    LOG(INFO) << "beta[" << sn << "]: " << beta;

    if(mdist > Dtype(1e-9)){
      LOG(INFO) << "sparse_gradient";

      caffe_cpu_strided_axpby(bottom[0]->num(), beta, diff_.cpu_data() + sn, sample_num_, 
        Dtype(0), bottom[0]->mutable_cpu_diff() + sn, sample_num_);
    }
    else{
      caffe_cpu_strided_axpby(bottom[0]->num(), Dtype(0), diff_.cpu_data() + sn, sample_num_, 
        Dtype(0), bottom[0]->mutable_cpu_diff() + sn, sample_num_);
      //for(int i = 0; i < bottom[0]->num(); i++){
      //  bottom[0]->mutable_cpu_diff()[ i * sample_num_ + sn ] = Dtype(0);
      //}
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SparseLossLayer);
#endif

INSTANTIATE_CLASS(SparseLossLayer);
REGISTER_LAYER_CLASS(SparseLoss);

}  // namespace caffe
