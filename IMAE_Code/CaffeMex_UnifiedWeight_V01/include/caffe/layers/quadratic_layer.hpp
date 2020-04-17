#ifndef CAFFE_QUADRATIC_LAYER_HPP_
#define CAFFE_QUADRATIC_LAYER_HPP_

#include <vector>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class QuadraticLayer : public Layer<Dtype> {
 public:
  explicit QuadraticLayer(const LayerParameter& param) 
      : Layer<Dtype>(param) {}
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>&top);
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QUADRATIC"; }
  
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

  inline Dtype QuadraticFunc(const Dtype x, const Dtype coef, const Dtype mean) 
  {
    Dtype a = Dtype(-1.0) * coef;

    return a * x * (x - 2*mean);
  }
  
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- QuadraticLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

};

}  // namespace caffe

#endif  // CAFFE_QUADRATIC_LAYER_HPP_
