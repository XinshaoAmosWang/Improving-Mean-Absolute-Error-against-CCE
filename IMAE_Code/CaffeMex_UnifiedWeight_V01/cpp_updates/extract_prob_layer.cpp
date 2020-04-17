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
}

template <typename Dtype>
void ExtractProbLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int sample_num = bottom[0]->num();
  int channels = bottom[0]->count() / bottom[0]->num();

  for(int num = 0; num < sample_num; num++)
  {
    int label = bottom_label[num];
    top_data[num] = bottom_data[num * channels + label];

  } 
  
}


#ifdef CPU_ONLY
STUB_GPU(ExtractProbLayer);
#endif

INSTANTIATE_CLASS(ExtractProbLayer);
REGISTER_LAYER_CLASS(ExtractProb);

}  // namespace caffe
