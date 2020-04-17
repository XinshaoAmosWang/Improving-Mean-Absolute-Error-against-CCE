#include <algorithm>
#include <cfloat>
#include <vector>
#include <boost/pointer_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/layers/distance_weighted_valid_pair_loss_V4_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DistanceWeightedValidPairLossV4Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  //auxiliary layer
  LayerParameter contrastive_loss_param(this->layer_param_);
  contrastive_loss_param.set_type("ContrastiveLoss");
  contrastive_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(contrastive_loss_param);

}

template <typename Dtype>
void DistanceWeightedValidPairLossV4Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::Reshape(bottom, top);
  //initialization of  pair data and pair lables, and auxiliary parameters
  sample_num = bottom[0]->num();
  feature_dim = bottom[0]->channels();
  //Valid construction of pairs
  pair_num = 0;
  pair_index_a.clear();
  pair_index_b.clear();
  pair_score.clear();
  for (int i = 0; i < sample_num - 1; i++){
    for(int j = i + 1; j < sample_num; j++){
      pair_num++;
      pair_index_a.push_back(i);
      pair_index_b.push_back(j);
      pair_score.push_back(Dtype(1.0));
    }
  }
  //
  vector<int> shape = bottom[0]->shape();
  shape[0] = pair_num;
  pair_data_a.Reshape(shape);
  pair_data_b.Reshape(shape);
  pair_label.Reshape(pair_num, 1, 1, 1);

  contrastive_loss_bottom.clear();
  contrastive_loss_bottom.push_back(&pair_data_a);
  contrastive_loss_bottom.push_back(&pair_data_b);
  contrastive_loss_bottom.push_back(&pair_label);
  
  contrastive_loss_top.clear();
  contrastive_loss_top.push_back(top[0]);

  contrastive_loss_layer_->SetUp(contrastive_loss_bottom, contrastive_loss_top);
  
  contrastive_propogate.clear();
  contrastive_propogate.push_back(true);
  contrastive_propogate.push_back(true);
  contrastive_propogate.push_back(false);  
  
}


template <typename Dtype>
void DistanceWeightedValidPairLossV4Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  //set the input data
  for (int i = 0; i < pair_num; i++){
    //The pair label
    if( bottom[1]->cpu_data()[pair_index_a[i]] == bottom[1]->cpu_data()[pair_index_b[i]] ){
      pair_label.mutable_cpu_data()[i] = Dtype(1);
    }
    else{
      pair_label.mutable_cpu_data()[i] = Dtype(0);
    }
    //the pair data
    const Dtype* source_a = bottom[0]->cpu_data() + pair_index_a[i] * feature_dim;
    const Dtype* source_b = bottom[0]->cpu_data() + pair_index_b[i] * feature_dim;
    Dtype* dest_a = pair_data_a.mutable_cpu_data() + i*feature_dim;
    Dtype* dest_b = pair_data_b.mutable_cpu_data() + i*feature_dim;

    caffe_copy(feature_dim, source_a, dest_a);
    caffe_copy(feature_dim, source_b, dest_b);
    //the pair score
    //pair_score[i] = bottom[2]->cpu_data()[pair_index_a[i]] * bottom[2]->cpu_data()[pair_index_b[i]];
    pair_score[i] = std::min(bottom[2]->cpu_data()[pair_index_a[i]] , bottom[2]->cpu_data()[pair_index_b[i]]);
  }
  // The forward pass of contrastive loss.
  contrastive_loss_layer_->Forward(contrastive_loss_bottom, contrastive_loss_top);
}

template <typename Dtype>
void DistanceWeightedValidPairLossV4Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    // The backward pass of contrastive loss.
    contrastive_loss_layer_->Backward(contrastive_loss_top, contrastive_propogate, contrastive_loss_bottom);
    // set diff to zero
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
    //dynamic cast
    shared_ptr<ContrastiveLossLayer<Dtype> > auxiLayer = boost::dynamic_pointer_cast<ContrastiveLossLayer<Dtype> >(contrastive_loss_layer_);
    if (!auxiLayer) {
      throw std::runtime_error("ContrastiveLossLayer dynamic pointer_cast failed");
    }

    //reweighting
    Dtype sum_similar_dw = Dtype(0);
    Dtype sum_dissimilar_dw = Dtype(0);
    Dtype margin = this->layer_param_.contrastive_loss_param().margin();
    
    for(int i = 0; i < pair_num; i++){
      //distance based weighting
      if( static_cast<int>(pair_label.cpu_data()[i]) ){
        //similar pairs
        Dtype dist = sqrt(auxiLayer->dist_sq_.cpu_data()[i]);

        sum_similar_dw += (dist * pair_score[i]);
      }
      else{
        //dissimilar pairs
        Dtype dist = sqrt(auxiLayer->dist_sq_.cpu_data()[i]);
        Dtype m_dist = std::max(Dtype(0), margin - dist);

        sum_dissimilar_dw += (m_dist * pair_score[i]);
      }
    }
    sum_similar_dw += static_cast<Dtype>(1e-12);
    sum_dissimilar_dw += static_cast<Dtype>(1e-12);

    //copy gradients and scale the gradients with normalized score
    for(int i = 0; i < pair_num; i++){
      //separate process for similar pairs and dissimilar pairs
      if( static_cast<int>(pair_label.cpu_data()[i]) ){
        //similar pairs
        Dtype dist = sqrt(auxiLayer->dist_sq_.cpu_data()[i]);

        pair_score[i] = 0.5 * (dist * pair_score[i]) / sum_similar_dw;
      }
      else{
        //dissimilar pairs
        Dtype dist = sqrt(auxiLayer->dist_sq_.cpu_data()[i]);
        Dtype m_dist = std::max(Dtype(0), margin - dist);

        pair_score[i] = 0.5 * (m_dist * pair_score[i]) / sum_dissimilar_dw;
      }
      //LOG(INFO) << "pair_score[" << i <<"]: " << pair_score[i];

      const Dtype* source_a = pair_data_a.cpu_diff() + i * feature_dim;
      const Dtype* source_b = pair_data_b.cpu_diff() + i * feature_dim;
      Dtype* dest_a = bottom[0]->mutable_cpu_diff() + pair_index_a[i] * feature_dim;
      Dtype* dest_b = bottom[0]->mutable_cpu_diff() + pair_index_b[i] * feature_dim;
      //dest[ind] = pair_score[i] * source[ind] + dest[ind]
      caffe_axpy(feature_dim, pair_score[i], source_a, dest_a);
      caffe_axpy(feature_dim, pair_score[i], source_b, dest_b);
    }     

  }
  if (propagate_down[1] || propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label and score inputs.";
  }

}

#ifdef CPU_ONLY
STUB_GPU(DistanceWeightedValidPairLossV4Layer);
#endif

INSTANTIATE_CLASS(DistanceWeightedValidPairLossV4Layer);
REGISTER_LAYER_CLASS(DistanceWeightedValidPairLossV4);

}  // namespace caffe