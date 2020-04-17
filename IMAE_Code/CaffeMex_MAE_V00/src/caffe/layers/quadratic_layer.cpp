#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/quadratic_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void QuadraticLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 1);
}

template <typename Dtype>
void QuadraticLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void QuadraticLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  Dtype coef = this->layer_param_.quadratic_param().coef();
  Dtype mean = this->layer_param_.quadratic_param().mean();

  for(int i = 0; i < bottom[0]->count(); i++ )
  {
    top_data[i] = QuadraticFunc(bottom_data[i], coef, mean);
  }
}


#ifdef CPU_ONLY
STUB_GPU(QuadraticLayer);
#endif

INSTANTIATE_CLASS(QuadraticLayer);
REGISTER_LAYER_CLASS(Quadratic);

}  // namespace caffe
