## IMAE for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters

* [Paper](https://arxiv.org/pdf/1903.12141.pdf)
* [Slide: Example Weighting for Deep Representation Learning using IMAE and DM](./IMAE_DM_V03_17052020_at_KAUST_VISION-CAIR_GROUP.pdf)
* Code is under legal check... 
* This work has contributed to my PhD thesis: [Example Weighting for Deep Representation Learning](https://pure.qub.ac.uk/en/studentTheses/example-weighting-for-deep-representation-learning). Therefore, this work passed the review/examination. Please feel safe to cite our work. 

#### :+1: Glad to know that our recent papers have inspired an ICML 2020 paper: [Normalized Loss Functions for Deep Learning with Noisy Labels](https://arxiv.org/pdf/2006.13554.pdf)
* [Open discussion on deep robustness, please!](https://www.reddit.com/r/MachineLearning/comments/hjlayq/r_open_discussion_on_deep_robustness_please/)
* This paper worked on improving loss functions. Insteadm, in our work, we go deeper and propose a much more flexible framework **to design the derivative straightforwardly without deriving it from a loss function, [is not it very cool?](https://arxiv.org/pdf/1905.11233.pdf)**
* Besides, we propose to **interpret the derivative magnitude of one data point as its weight. Rethinking of deep learning optimisation is probably needed [here](https://arxiv.org/pdf/1905.11233.pdf)!**

<!--
#### :+1: [Code releasing](https://xinshaoamoswang.github.io/blogs/2020-02-18-code-releasing/)
-->


#### :+1: Selected work partially impacted by our work
* [ICML-20: Normalized Loss Functions for Deep Learning with Noisy Labels](https://arxiv.org/pdf/2006.13554.pdf)
* [ICML-20: SIGUA: Forgetting May Make Learning with Noisy Labels More Robust](https://proceedings.icml.cc/static/paper_files/icml/2020/705-Paper.pdf)
  * [Notes and remarks](https://xinshaoamoswang.github.io/blogs/2020-06-14-Robust-Deep-LearningviaDerivativeManipulationIMAE/#how-do-you-think-of-requesting-kind-citations)

* [NeurIPS-20: Early-Learning Regularization Prevents Memorization of Noisy Labels](https://proceedings.neurips.cc/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html)
  * **The analysis about "gradient and example weighting" has been done in our [IMAE](https://arxiv.org/abs/1903.12141) + [DM](https://arxiv.org/abs/1905.11233)**, which mathematically prove that CCE tends to over-fit and why, and how to propose robust example weighting schemes.
  * Their analysis in Page#4: During the early-learning stage, the algorithm makes progress and the accuracy on wrongly labeledexamples increases. However, during this initial stage, the relative importance of the wrongly labeled examples continues to grow; once the effect of the wrongly labeled examples begins to dominate, memorization occurs.

* [NeurIPS-20: Coresets for Robust Training of Deep Neural Networks against Noisy Labels](https://proceedings.neurips.cc/paper/2020/hash/8493eeaccb772c0878f99d60a0bd2bb3-Abstract.html)
  * The key idea behind this method is to select subsets of clean data points that provide an approximately low-rank Jacobian matrix. The authors then prove that gradient descent applied to the subsets cannot overfit the noisy labels, even without regularization or early stopping.

* [Medical Image Analysis: Deep learning with noisy labels: Exploring techniques and remedies in medical image analysis](https://www.sciencedirect.com/science/article/pii/S1361841520301237)

* [2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI): Learning to Detect Brain Lesions from Noisy Annotations](https://ieeexplore.ieee.org/abstract/document/9098599)
* [A Survey on Deep Learning with Noisy Labels: How to train your model when you cannot trust on the annotations?](http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2020/09.30.23.54/doc/Tutorial_ID_4_SIBGRAPI_2020_camara_ready_v2%20copy.pdf)

#### Please check our following work at [Derivative Manipulation for General Example Weighting, May 2019](https://xinshaoamoswang.github.io/blogs/2020-06-14-Robust-Deep-LearningviaDerivativeManipulationIMAE/). 
 


## Citation

Please kindly cite us if you find our work useful.

```
@article{wang2019imae,
  title={{IMAE} for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Robertson, Neil M},
  journal={arXiv preprint arXiv:1903.12141},
  year={2019}
}
```


## Open Reviews and Discussion 

**Since this paper is released, for your better reference, the ICCV-19 reviews results are released following the practice of OpenReview**
* [Reviews](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE/blob/master/ICCV19_FinalReviewsRejected/Conference%20Management%20Toolkit%20-%20View%20review.pdf)
* [RebuttalToReviewers](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE/blob/master/ICCV19_FinalReviewsRejected/IMAE_rebuttals_V07.pdf)
& [RebuttalToAC](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE/blob/master/ICCV19_FinalReviewsRejected/Conference%20Management%20Toolkit%20-%20View%20Author%20Feedback.pdf)
* [Meta-Reviews](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE/blob/master/ICCV19_FinalReviewsRejected/Conference%20Management%20Toolkit%20-%20View%20meta-review.pdf)

* [More Discussions](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE#discussion)

**A Open Question on whether clean or noisy validation set for ML/DL researchers caring about label noise**

* [Reviewer#3's opinion in final justification](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE/blob/master/ICCV19_FinalReviewsRejected/Conference%20Management%20Toolkit%20-%20View%20review.pdf): `The validation sets are required to be clean, which greatly decrease the contribution. **Many existing methods
employ noisy validation set to choose hyper-parameters**, e.g., when the risk is consistent. **As minimizing risks on
the noisy validation set is asymptotically equal to minimizing risk on the clean data**.'

* [My opinion discussed with my collaborators](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE/blob/master/ICCV19_FinalReviewsRejected/IMAE_rebuttals_V07.pdf): Following the ML literature, **a validation set should be clean** as
**we should not expect a ML model to predict noisy data well**.
In other words, **we cannot evaluate/decide a model’s performance
on noisy validation/test data**. Our goal is to avoid learning
faults from noisy data and generalise better during inference.

**Positive comments we collected**
* The proposed modification IMAE is **quite simple and should be considerably more efficient than other methods that deal with label noise**.
* The theoretical analysis of CCE and MAE is thorough and provides an explanation of the tendency of CCE to
overfit to incorrect labels and the underfitting of MAE to the correct labels. 
* The experiments show a significant improvement over CCE in the case of noisy labels which validates the
approach. I also appreciate the experiment on MARS with realistic label noise. I appreciate the comparison on Clothing-1M provided by the authors. The results there suggest that
under realistic label noise the method actually works well when compared to SotA methods.

<!--
**What next?**
* We will improve our work based on the ICCV-19 reviews, e.g., adding more experiments. -->

## Introduction

**Research questions:**
* Why does MAE work much worse than CCE although it is noise-robust?
* How to improve MAE against CCE to embrace noise-robustness and high generalisation performance?

Our work is a further study of robust losses following MAE [1] and GCE [2]. They proved MAE is more robust than CCE when noise exists. However, MAE’s underfitting phenomenon is not exposed and studied in the literature. We analysed it thoroughly and proposed a simple solution to embrace both high fitting ability (accurate) and test stability (robust). 

<!--
**Our main purpose is not a proposal to push current best performance under label noise.** Instead, we focus on analysing how different losses perform differently and why, which is a fundamental research question. 

Our focus is to analyse why CCE overfits while MAE underfits as presented in ablation studies in Table 2. Under unknown real-world noise in Table 3, we only compared with GCE [2] as it is the most related and demonstrated to be the state-of-the-art.
-->

**IMAE is suitable for cases where inputs and labels may be unmatched.**

Training DNNs requires rethinking data fitting and generalisation.
**Our main contribution is simple analysis and solution from the viewpoint of gradient magnitude with respect to logits.** 

## Takeaways


<p float="left">
  <img src="./fig/illustration_MAE_IMAE_CCE.png" width="400">
  <img src="./fig/illustration_MAE_IMAE_CCE_caption.png" width="400">
</p>

<p float="left">
  <img src="./fig/introduction_table.png" width="400">
  <img src="./fig/introduction_caption.png" width="400">
</p>

<p float="left">
  <img src="./fig/table2_caption.png" width="800">
  <img src="./fig/table2.png" width="800">
</p>

<p float="left">
  <img src="./fig/figure3.png" width="400">
  <img src="./fig/figure3_caption.png" width="400">
</p>


* By ‘CCE is noise-sensitive and overfits training data’, we mean CCE owns high data fitting accuracy but its final test accuracy drops a lot versus its best test accuracy.
* By ‘MAE is robust’, we mean MAE’s final test accuracy drops only a bit versus its best test one.
* By ‘MAE underfits training data’, we mean its training and best test accuracies are low.

Please see our empirical evidences in the paper.


**MAE’s fitting ability is much worse than CCE. In other words, CCE overfits to incorrect labels while MAE underfits to correct labels.**
* **The robustness/sensitive to noise is from the angle of test accuracy stability/trend**, i.e., CCE’s final
test accuracy drops a lot versus its best one while MAE’s
final one is almost the same as its best one; 
* **The claim ‘MAE works worse than CCE’ is from the aspect of best test accuracy** since we generally apply early stopping to help
CCE. 

## Results

**Label noise is one of the most explicit cases where some observations and their labels are not matched in the training data. In this case, it is quite crucial to make your models learn meaningful patterns instead of errors.**

### Synthetic noise

<p float="left">
  <img src="./fig/table3.png" width="400">
  <img src="./fig/table3_caption.png" width="400">
</p>

<p float="left">
  <img src="./fig/table4.png" width="400">
</p>



<img src="./fig/table6.png" width="800">
<img src="./fig/figure4.png" width="800">


### Real-world unknown noise

**Classification on Clothing 1M [a] is here**

<img src="./fig/table5.png" width="800">


## Hyper-paramter Analysis 

<img src="./fig/illustration_IMAE.png" width="400">

<img src="./fig/train_dynamics_T.png" width="800">

<img src="./fig/test_dynamics_T.png" width="800">

## Discussion

#### 1.  The idea of this paper is quite close to "training deep neural-networks using a noise adaptation layer"? They both intend to change the weight of each sample before sending to softmax, definitely they do in different ways. It decreases the novelty of this paper?

Their critical differences are: 1) Noise Adaption explicitly estimates
latent true labels by an additional softmax layer while our
IMAE reweights examples based on their input-to-label relevance
scores; 2) IMAE reweights samples **after softmax**,
i.e., scaling their gradients as shown in Eq. (22) in our paper.

#### 2. Why uniform noise (symmetric/class-independent noise )?
We choose uniform noise because it is more challenging than
asymmetric (class-dependent) noise which was verified in [d] Vahdat et al. Toward robustness against label noise in training
deep discriminative neural networks. In NeurIPS, 2017.

#### 3. Why is the performance still okay when noise rate is 80%?
By adding uniform noise, **even up to 80%, the correct portion is still the majority**, since the 80% are relocated to other 9 classes evenly.

Being natural and intuitive, the majority voting
decides the meaningful data patterns to learn. We believe that if the
noise accounts the majority, DNNs is hard to learn meaningful
patterns. Therefore, **the majority voting is our reasonable assumption.**

#### 4. The study from the gradient perspective is not new, e.g., Truncated Cauchy Non-Negative Matrix Factorization, ang GCE [2].  

Yes, we agree the perspective itself is not new. However,
we find how we analyse fundamentally and go to the simple
solution via the gradient viewpoint is novel.

Truncated Cauchy Non-Negative Matrix Factorization (TPAMI-2017) and GCE [2] truncate large errors to filter out extreme outliers. Instead, our
IMAE adjusts weighting variance without dropping any samples.

#### 5. The robustness is not specific for label noise. I think the method works well for general noise, e.g., outliers.

Yes, that is a great point. Our IMAE is suitable for all cases
where inputs and their labels are not semantically matched,
which may come from noisy data or labels. Since we only
evaluated on label noise, we did not exaggerate its efficacy.

We will test more cases in the future. 

#### 6. Is the validation data clean or not? If clean, this would greatly reduce the contribution of the paper.

Following the ML literature, a validation set should be clean as
we should not expect a ML model to predict noisy data well.
In other words, we cannot evaluate a model’s performance
on noisy validation/test data. Our goal is to avoid learning
faults from noisy data and generalise better during inference.

#### 7. More experiments with comparison to prior work and more evaluation on real-world datasets with unknown noise? 

Our focus is to analyse why CCE
overfits while MAE underfits as presented in ablation studies
in Table 2. Under unknown real-world noise in Table 3, we
only compared with GCE [2] as it is the most related and
demonstrated to be the state-of-the-art.

**Classification on Clothing 1M [a] is here**

<img src="./fig/table5.png" width="800">




## References
[1] A. Ghosh, H. Kumar, and P. Sastry. Robust loss functions
under label noise for deep neural networks. In AAAI, 2017.

[2] Z. Zhang and M. R. Sabuncu. Generalized cross entropy loss
for training deep neural networks with noisy labels. In NeurIPS 2018.

[3] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals.
Understanding deep learning requires rethinking generalization.
In ICLR, 2017.

[4] L. Zheng, Z. Bie, Y. Sun, J. Wang, C. Su, S. Wang, and
Q. Tian. Mars: A video benchmark for large-scale person
re-identification. In ECCV, 2016.

[a] Xiao et al. Learning From Massive Noisy Labeled Data for
Image Classification. In CVPR, 2015.

[b] Patrini et al. Making deep neural networks robust to label noise:
A loss correction approach. In CVPR, 2017.

[c] Goldberger et al. Training deep neural-networks using a noise
adaptation layer. In ICLR, 2017.

[d] Vahdat et al. Toward robustness against label noise in training
deep discriminative neural networks. In NeurIPS, 2017.

[e] Tanaka et al. Joint optimization framework for learning with
noisy labels. In CVPR, 2018.

[f] Han et al. Masking: A new perspective of noisy supervision. In
NeurIPS, 2018.

[g] Jenni et al. Deep bilevel learning. In ECCV, 2018.
