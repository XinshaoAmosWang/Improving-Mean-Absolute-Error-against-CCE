#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/gaussian_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void GaussianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 1);
}

template <typename Dtype>
void GaussianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void GaussianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  Dtype sigma = this->layer_param_.gaussian_param().sigma();
  Dtype mean = this->layer_param_.gaussian_param().mean();

  //LOG(INFO) << "gaussianfunc: " << sigma << " " << mean; 
  for(int i = 0; i < bottom[0]->count(); i++ )
  {
    top_data[i] =  gaussianfunc(bottom_data[i], sigma, mean);
    //LOG(INFO) << bottom_data[i] << " " << top_data[i];
  }
}


#ifdef CPU_ONLY
STUB_GPU(GaussianLayer);
#endif

INSTANTIATE_CLASS(GaussianLayer);
REGISTER_LAYER_CLASS(Gaussian);

}  // namespace caffe
