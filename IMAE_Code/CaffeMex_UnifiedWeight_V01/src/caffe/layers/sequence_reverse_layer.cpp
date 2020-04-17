#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sequence_reverse_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SeqreverseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
}

template <typename Dtype>
void SeqreverseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  std::vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
  
  T_ = shape[0];
  inner_dim_ = bottom[0]->count() / T_;
}

template <typename Dtype>
void SeqreverseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for(int t1 = 0; t1 < T_; t1++ )
  {
    int t2 = (T_-1) - t1;
    caffe_copy<Dtype>(inner_dim_, 
                      bottom_data + (t1 * inner_dim_),
                      top_data + (t2 * inner_dim_) );
  }
}
template <typename Dtype>
void SeqreverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    for(int t1 = 0; t1 < T_; t1++ )
    {
      int t2 = (T_-1) - t1;
      caffe_copy<Dtype>(inner_dim_, 
                        top_diff + (t1 * inner_dim_),
                        bottom_diff + (t2 * inner_dim_) );
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SeqreverseLayer);
#endif

INSTANTIATE_CLASS(SeqreverseLayer);
REGISTER_LAYER_CLASS(Seqreverse);

}  // namespace caffe
