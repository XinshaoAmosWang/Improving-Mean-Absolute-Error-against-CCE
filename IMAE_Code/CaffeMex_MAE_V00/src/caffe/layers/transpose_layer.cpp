#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/transpose_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void TransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 1);
}

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  batch_size = bottom[0]->num();
  channels = bottom[0]->channels();
  inner_num = bottom[0]->count()/(batch_size*channels);
  //LOG(INFO) << "batch_size: " << batch_size;
  //LOG(INFO) << "channels: " << channels;
  //LOG(INFO) << "inner_num: " << inner_num;

  top[0]->Reshape(batch_size, inner_num, channels, 1);

}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for(int num = 0; num < batch_size; num++ )
  {
    //transpose
    for(int row = 0; row < inner_num; row++) //49
    {
      for(int col =0; col < channels; col++) //1024
      {
        top_data[num*channels*inner_num + row*channels + col] = bottom_data[num*channels*inner_num + col*inner_num + row];
      }
    }
    
  }
}


#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
