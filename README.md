# FeCAM [![Paper](https://img.shields.io/badge/arXiv-2210.07207-brightgreen)](https://arxiv.org/abs/2309.14062)
## Code for NeurIPS 2023 paper - FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning

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

## For many-shot CIL experiments

The framework for many-shot CIL setting is taken from [PyCIL](https://github.com/G-U-N/PyCIL).

### Dependencies
1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)

### Datasets

We performed experiments for `CIFAR100`, `ImageNet100,` and `TinyImageNet`. When training on `CIFAR100`, this framework will automatically download it.  When training on `ImageNet100` or `TinyImageNet`, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```

### Run experiment

1. Edit the exps/[Model name].json to change the experiment settings.
2. Run the following command for FeCAM
   
   ```
    python main.py --config==exps/fecam.json
   ```
3. Hyperparameters:
  - **memory-size**: The total exemplar number in the incremental learning process. We do not need to store exemplars for FecAM.
  - **init-cls**: The number of classes in the first incremental stage. 
  - **increment**: The number of classes in each incremental stage. 
  - **convnet-type**: The backbone network for the incremental model. We use `ResNet18` for all the experiments .
  - **seed**: The random seed adopted for shuffling the class order. According to the benchmark setting of PyCIL, it is set to 1993 by default.
  - **beta**: The degree of feature transformation using Tukey’s Ladder of Powers Transformation.
  - **alpha1, alpha2**: The hyperparameters for covariance shrinkage.

Other algorithm-specific hyperparameters can be modified in the corresponding json files. There are options to use NCM Classifier instead of FeCAM.

## To use FeCAM with pre-trained visual transformers

### Dependencies
1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [numpy](https://github.com/numpy/numpy)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [timm](https://pypi.org/project/timm/)
6. [continuum](https://pypi.org/project/continuum/)
   
### Datasets

Download the ImageNet-R and CoRe50 datasets.

### Run experiment

1. Run the following command:

```
python FeCAM_vit_{dataset}.py
```
2. The hyperparameters can be modified in the corresponding python files. To try the NCM classifier, run the following command:

```
python NCM_vit_{dataset}.py
```

## For few-shot CIL experiments

Code for the few-shot CIL experiments will be available soon.

