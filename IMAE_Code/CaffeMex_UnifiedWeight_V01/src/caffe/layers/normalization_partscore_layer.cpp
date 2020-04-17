#include <vector>
#include <cmath>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalization_partscore_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void NormalizationPartScoreLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) 
	{
		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void NormalizationPartScoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) 
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		
		int batch_num = bottom[0]->num(); //the batch number
		int elem_num = bottom[0]->count() / batch_num; // element number for each batch
		for (int batch_id = 0; batch_id < batch_num; ++batch_id) 
		{
			Dtype normabs = caffe_cpu_asum<Dtype>(elem_num, bottom_data + batch_id*elem_num) + static_cast<Dtype>(1e-12);

			caffe_cpu_scale<Dtype>(elem_num, pow(normabs, -1), bottom_data + batch_id*elem_num, top_data + batch_id*elem_num);
		}
	}

	template <typename Dtype>
	void NormalizationPartScoreLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
	{
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* top_data = top[0]->cpu_data();

		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		

		int batch_num = bottom[0]->num(); //the batch number
		int elem_num = bottom[0]->count() / batch_num; // element number for each batch
		for (int batch_id = 0; batch_id < batch_num; ++batch_id) 
		{
			Dtype sum_grad = caffe_cpu_dot(elem_num, top_data + batch_id*elem_num, top_diff + batch_id*elem_num);
			
			caffe_set(elem_num, sum_grad, bottom_diff + batch_id*elem_num);
			caffe_sub(elem_num, top_diff + batch_id*elem_num, bottom_diff + batch_id*elem_num, bottom_diff + batch_id*elem_num);

			Dtype normabs = caffe_cpu_asum<Dtype>(elem_num, bottom_data + batch_id*elem_num)+ static_cast<Dtype>(1e-12);
			caffe_cpu_scale(elem_num, Dtype(pow(normabs, -1)), bottom_diff + batch_id*elem_num, bottom_diff + batch_id*elem_num);
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(NormalizationPartScoreLayer);
#endif

	INSTANTIATE_CLASS(NormalizationPartScoreLayer);
	REGISTER_LAYER_CLASS(NormalizationPartScore);

}  // namespace caffe