#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/circle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CircleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 1);
}

template <typename Dtype>
void CircleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void CircleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  Dtype mean = this->layer_param_.circle_param().mean();

  for(int i = 0; i < bottom[0]->count(); i++ )
  {
    top_data[i] = CircleFunc(bottom_data[i], mean);
  }
}


#ifdef CPU_ONLY
STUB_GPU(CircleLayer);
#endif

INSTANTIATE_CLASS(CircleLayer);
REGISTER_LAYER_CLASS(Circle);

}  // namespace caffe
