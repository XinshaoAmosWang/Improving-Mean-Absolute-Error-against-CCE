#ifndef CAFFE_GAUSSIAN_LAYER_HPP_
#define CAFFE_GAUSSIAN_LAYER_HPP_

#include <vector>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GaussianLayer : public Layer<Dtype> {
 public:
  explicit GaussianLayer(const LayerParameter& param) 
      : Layer<Dtype>(param) {}
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>&top);
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GAUSSIAN"; }
  
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

  inline Dtype gaussianfunc(const Dtype x, const Dtype sigma, const Dtype u) 
  {
    //Dtype u = Dtype(0.5);
    //Dtype sigma = Dtype(0.18);

    Dtype expVal = Dtype(-0.5) * pow( (x - u) / sigma, 2);

    Dtype divider = sqrt(2 * M_PI * pow(sigma, 2));
    
    return (Dtype(1) / divider) * exp(expVal);
  }
  
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- GaussianLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

};

}  // namespace caffe

#endif  // CAFFE_GAUSSIAN_LAYER_HPP_
