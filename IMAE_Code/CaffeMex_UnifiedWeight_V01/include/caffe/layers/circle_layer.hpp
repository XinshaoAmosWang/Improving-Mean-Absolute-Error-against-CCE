#ifndef CAFFE_CIRCLE_LAYER_HPP_
#define CAFFE_CIRCLE_LAYER_HPP_

#include <vector>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CircleLayer : public Layer<Dtype> {
 public:
  explicit CircleLayer(const LayerParameter& param) 
      : Layer<Dtype>(param) {}
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>&top);
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CIRCLE"; }
  
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

  inline Dtype CircleFunc(const Dtype x, const Dtype mean) 
  {
    if (x >= mean)
      return ( mean- pow(mean*mean - (x-2*mean)*(x-2*mean), 0.5) );
    else
      return ( mean- pow(mean*mean - x*x, 0.5) );
  }
  
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- CircleLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

};

}  // namespace caffe

#endif  // CAFFE_CIRCLE_LAYER_HPP_
