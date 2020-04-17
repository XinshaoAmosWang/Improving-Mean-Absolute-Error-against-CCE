#ifndef CAFFE_EXTRACT_PROB_LAYER_HPP_
#define CAFFE_EXTRACT_PROB_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ExtractProbLayer : public Layer<Dtype> {
 public:
  explicit ExtractProbLayer(const LayerParameter& param) 
      : Layer<Dtype>(param) {}
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>&top);
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ExtractProb"; }
  
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- ExtractProbLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

};

}  // namespace caffe

#endif  // CAFFE_EXTRACT_PROB_LAYER_HPP_
