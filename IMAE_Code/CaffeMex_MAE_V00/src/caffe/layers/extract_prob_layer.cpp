#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/extract_prob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ExtractProbLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 2);
}

template <typename Dtype>
void ExtractProbLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> shape = bottom[0]->shape();
  
  shape[1] = 1;
  top[0]->Reshape(shape);

  if(top.size() >=2 )
  {
    top[1]->Reshape(1,1,1,1);//store the max_prob_tracklet
  }


  batch_size = bottom[0]->num();
  channels = bottom[0]->channels();
  inner_num = bottom[0]->count()/(batch_size*channels);
  //LOG(INFO) << "batch_size: " << batch_size;
  //LOG(INFO) << "channels: " << channels;
  //LOG(INFO) << "inner_num: " << inner_num;
}

template <typename Dtype>
void ExtractProbLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  //Only process one tracklet
  Dtype max_prob_tracklet = Dtype(0);
  for(int num = 0; num < batch_size; num++ )
  {
    int label = static_cast<int>( bottom_label[num] );
    //LOG(INFO) << "label[" << num << "]: " << label;
    //
    for(int i = 0; i < inner_num; i++ )
    {
      top_data[num*inner_num + i] = bottom_data[num*channels*inner_num + label*inner_num + i];
      //
    }
    //LOG(INFO) <<"prob[" << num << "*" << inner_num << "+" << 0 <<"]: " << top_data[num*inner_num + 0];  
    //only consider the first/inner_num: i = 0
    max_prob_tracklet = max(max_prob_tracklet, top_data[num*inner_num + 0]);
  }

  if(top.size() >= 2)
  {
    top[1]->mutable_cpu_data()[0] = max_prob_tracklet;
    LOG(INFO) << "max_prob_tracklet: " << max_prob_tracklet;
  }
  
}


#ifdef CPU_ONLY
STUB_GPU(ExtractProbLayer);
#endif

INSTANTIATE_CLASS(ExtractProbLayer);
REGISTER_LAYER_CLASS(ExtractProb);

}  // namespace caffe
