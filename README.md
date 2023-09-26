# FeCAM
## Code for NeurIPS 2023 paper - FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning [![Paper](https://img.shields.io/badge/arXiv-2210.07207-brightgreen)](https://arxiv.org/abs/2309.14062)

## Abstract
Exemplar-free class-incremental learning (CIL) poses several challenges since it prohibits the rehearsal of data from previous tasks and thus suffers from catastrophic forgetting. Recent approaches to incrementally learning the classifier by freezing the feature extractor after the first task have gained much attention. In this paper, we explore prototypical networks for CIL, which generate new class prototypes using the frozen feature extractor and classify the features based on the Euclidean distance to the prototypes. In an analysis of the feature distributions of classes, we show that classification based on Euclidean metrics is successful for jointly trained features. However, when learning from non-stationary data, we observe that the Euclidean metric is suboptimal and that feature distributions are heterogeneous. To address this challenge, we revisit the anisotropic Mahalanobis distance for CIL. In addition, we empirically show that modeling the feature covariance relations is better than previous attempts at sampling features from normal distributions and training a linear classifier. Unlike existing methods, our approach generalizes to both many- and few-shot CIL settings, as well as to domain-incremental settings. Interestingly, without updating the backbone network, our method obtains state-of-the-art results on several standard continual learning benchmarks.

```
@inproceedings{goswami2023fecam,
  title={FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning}, 
  author={Dipam Goswami and Yuyang Liu and Bartłomiej Twardowski and Joost van de Weijer},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Dependencies
1. torch 1.81
2. torchvision 0.6.0
3. tqdm
4. numpy
5. scipy

## To run the experiments
1. Edit the exps/[Model name].json to change the experiment settings.
2. Run
   ` python main.py --config==exps/fecam.json`



This repository is a modified version of [PyCIL](https://github.com/G-U-N/PyCIL).
