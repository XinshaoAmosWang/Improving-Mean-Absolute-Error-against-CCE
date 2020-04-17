#include <algorithm>
#include <cfloat>
#include <vector>
#include <boost/pointer_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "caffe/layers/weighted_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedContrastiveLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  //auxiliary layer
  LayerParameter contrastive_loss_param(this->layer_param_);
  contrastive_loss_param.set_type("ContrastiveLoss");
  contrastive_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(contrastive_loss_param);

}

template <typename Dtype>
void WeightedContrastiveLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::Reshape(bottom, top);
  //initialization of  pair data and pair lables, and auxiliary parameters
  sample_num = bottom[0]->num();
  feature_dim = bottom[0]->channels();
  //
  pair_num = 0;
  pair_index_a.clear();
  pair_index_b.clear();
  pair_score.clear();
  for (int i = 0; i < sample_num; i++){
    for(int j = 0; j < sample_num; j++){
      if(j != i){
        pair_num++;
        pair_index_a.push_back(i);
        pair_index_b.push_back(j);
        pair_score.push_back(Dtype(1.0));
      }
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
void WeightedContrastiveLossLayer<Dtype>::Forward_cpu(
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
void WeightedContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

    //Hard Mining of valid pairs and valid pair scores
    hardest_index_vec.clear();
    sum_score = Dtype(0);
    //
    int pair_index = -1;
    for (int i = 0; i < sample_num; i++){
    	Dtype positive_max_distance = Dtype(0);
    	//Dtype negative_min_distance = Dtype(2);
    	Dtype negative_min_distance = Dtype(4);
    	int positive_pair_index = -1;
    	int negative_pair_index = -1;
    	for (int j = 0; j < sample_num; j++){
    		if(j != i){
	          pair_index++;
	        
	          if( static_cast<int>(pair_label.cpu_data()[pair_index]) ){
	            //process similiar pairs: positives
	            if(auxiLayer->dist_sq_.cpu_data()[pair_index] > positive_max_distance){
	              positive_max_distance = auxiLayer->dist_sq_.cpu_data()[pair_index];
	              positive_pair_index = pair_index;
	            }
	          }
	          else{
	            //process dissimiliar pairs: negatives
	            if(auxiLayer->dist_sq_.cpu_data()[pair_index] < negative_min_distance){
	              negative_min_distance = auxiLayer->dist_sq_.cpu_data()[pair_index];
	              negative_pair_index = pair_index;
	            }
	          }
	        }
    	}
    	if (positive_pair_index != -1){
	        hardest_index_vec.push_back(positive_pair_index);
	        
	        sum_score += pair_score[positive_pair_index];
    	}
    	if (negative_pair_index != -1){
	        hardest_index_vec.push_back(negative_pair_index);

	        sum_score += pair_score[negative_pair_index];
    	}
    }
    sum_score += static_cast<Dtype>(1e-12);

    //copy gradients and scale the gradients with normalized score
    for (int valid_ind = 0; valid_ind < hardest_index_vec.size(); valid_ind++ )
    {
      //i-th pair
      int i = hardest_index_vec[valid_ind];   
      pair_score[i] = pair_score[i] / sum_score;
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
STUB_GPU(WeightedContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(WeightedContrastiveLossLayer);
REGISTER_LAYER_CLASS(WeightedContrastiveLoss);

}  // namespace caffe