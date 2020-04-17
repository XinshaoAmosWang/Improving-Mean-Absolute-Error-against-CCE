#ifndef CAFFE_OHPM_WEIGHTED_PAIR_LOSS_LAYER_HPP_
#define CAFFE_OHPM_WEIGHTED_PAIR_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/contrastive_loss_layer.hpp"

namespace caffe {

/**
 * @brief  *
 * Author: Anonymous
 * Email: Anonymous@gmail.com
 * Time: 04/2018
 * OHPM: Online Hard Pair Mining
 * @param bottom input Blob vector (length 3)
 *   -# @f$ (N \times C \times 1 \times 1) @f$
 *      the features
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the labels
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the scores
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the loss
 */
template <typename Dtype>
class OHPMWeightedPairLossLayer : public LossLayer<Dtype> {
 public:
   /**
    * @param param provides LossParameter loss_param, with options:
    *   
    */
  explicit OHPMWeightedPairLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OHPMWeightedPairLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  /**
   * We usually cannot backpropagate to the labels, scores; 
   * ignore force_backward for these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return !(bottom_index >= 1);
  }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  //vector<shared_ptr<Layer<Dtype> > > contrastive_loss_layer_vec_;
  
  //auxiliary layer
  shared_ptr<Layer<Dtype> >  contrastive_loss_layer_;
  vector<Blob<Dtype>*> contrastive_loss_bottom;
  vector<Blob<Dtype>*> contrastive_loss_top;
  vector<bool> contrastive_propogate;

  //auxiliary index for link: the length of a and b is the same;
  vector<int> pair_index_a;
  vector<int> pair_index_b;
  vector<Dtype> pair_score;
  //auxiliary parameters for the auxiliary layer
  Blob<Dtype> pair_data_a;
  Blob<Dtype> pair_data_b;
  Blob<Dtype> pair_label; 
  int sample_num;
  int pair_num;
  int feature_dim;
  Dtype sum_score;

  vector<int> hardest_index_vec;  
};

}  // namespace caffe

#endif  // CAFFE_OHPM_WEIGHTED_PAIR_LOSS_LAYER_HPP_