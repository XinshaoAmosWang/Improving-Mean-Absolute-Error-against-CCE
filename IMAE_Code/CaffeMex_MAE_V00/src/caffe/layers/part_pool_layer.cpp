#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/part_pool_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PartPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 2);
  
  has_weight = bottom.size() == 2 ? 1 : 0;
  
  batch_size_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  

}

template <typename Dtype>
void PartPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  std::vector<int> shape = bottom[0]->shape();
  shape[2] = 1;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void PartPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  //the element number of top[0]
  int slice_cnt = top[0]->count();  //b x 1024
  caffe_set(slice_cnt, Dtype(0.0), top_data);
    
  if (has_weight)
  {
    const Dtype* weight = bottom[1]->cpu_data();

    for(int num_b = 0; num_b < batch_size_; num_b++)
    {
      for(int j = 0; j<channels_; j++)
      {
        for(int num_h = 0; num_h < height_; num_h++)
        {
          top_data[num_b*channels_ + j] += bottom_data[num_b*channels_*height_ + j*height_ + num_h] * weight[num_b*height_ + num_h];
        }
      }
    }
    LOG(INFO) << "weight[" << 0 << "*"<< 0 << "+" << 0 << "]: " << weight[0*0 + 0];
    LOG(INFO) << "weight[" << 0 << "*"<< 0 << "+" << 1 << "]: " << weight[0*0 + 1];
    LOG(INFO) << "weight[" << 0 << "*"<< 0 << "+" << 2 << "]: " << weight[0*0 + 2];

  }
  else
  {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void PartPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  
  if (has_weight){
    
    const Dtype* weight = bottom[1]->cpu_data();
    Dtype* weight_diff = bottom[1]->mutable_cpu_diff();
    
    // diff for input
    for (int num_b = 0; num_b < batch_size_ ; ++num_b)
    {
        for(int num_h =0; num_h < height_; ++num_h)
        {
          //On return, the contents of vector Y are replaced with the result. 
          //The value computed is (alpha * X[i]) + Y[i].
          caffe_strided_axpy(channels_, weight[num_b*height_ + num_h], 
            top_diff + (num_b*channels_), 1, 
            bottom_diff + (num_b*channels_*height_) + num_h , height_ );
        }
    }

    // diff for weight
    for (int num_b = 0; num_b < batch_size_; ++num_b)
    {
      for(int num_h = 0; num_h < height_; ++num_h)
      {
        weight_diff[num_b*height_ + num_h] = caffe_cpu_strided_dot(channels_, 
          bottom_data + (num_b*channels_*height_) + num_h , height_ , 
          top_diff + (num_b*channels_), 1);
        /*
        //weight_diff[num_b*height_ + num_h] = weight_diff[num_b*height_ + num_h] / Dtype(channels_);
        */
        //LOG(INFO) << "weight_diff[" << num_b << "*"<< height_ << "+" << num_h << "]: " << weight_diff[num_b*height_ + num_h];

      }
    }
  }
  else
  {
    NOT_IMPLEMENTED;
  }
}

#ifdef CPU_ONLY
STUB_GPU(PartPoolLayer);
#endif

INSTANTIATE_CLASS(PartPoolLayer);
REGISTER_LAYER_CLASS(PartPool);

}  // namespace caffe
