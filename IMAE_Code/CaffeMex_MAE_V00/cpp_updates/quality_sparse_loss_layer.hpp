#ifndef CAFFE_QUALITY_SPARSE_LOSS_LAYER_HPP_
#define CAFFE_QUALITY_SPARSE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 */
template <typename Dtype>
class SparseLossLayer : public LossLayer<Dtype> {
//class SparseLossLayer : public LossLayer<Dtype> {
 public:
  explicit SparseLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {}
  //explicit SparseLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SparseLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
  /* *
  * * In the SparseLossLayer we can backpropagate
  * * to the first inputs.
  * */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

 protected:
  /// @copydoc SparseLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  Blob<Dtype> diff_;
  Blob<Dtype> weights_ave_;
  Dtype margin;
  Dtype loss;
  Blob<Dtype> dist_sq_;  // cached for backward pass

};

}  // namespace caffe

#endif  // CAFFE_QUALITY_SPARSE_LOSS_LAYER_HPP_
