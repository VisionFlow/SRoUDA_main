# Unsupervised Domain Adaptation for Image Classification

## Installation
Example scripts can deal with [WILDS datasets](https://wilds.stanford.edu/).
You should first install ``wilds`` before using these scripts.

```
pip install wilds
```

Example scripts also support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models).
You also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [MNIST](http://yann.lecun.com/exdb/mnist/), [SVHN](http://ufldl.stanford.edu/housenumbers/), [USPS](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps) 
- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeCaltech](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [VisDA2017](http://ai.bu.edu/visda-2017/)
- [DomainNet](http://ai.bu.edu/M3SDA/)
- [iwildcam (WILDS)](https://wilds.stanford.edu/datasets/)
- [camelyon17 (WILDS)](https://wilds.stanford.edu/datasets/)
- [fmow (WILDS)](https://wilds.stanford.edu/datasets/)

You need to prepare following datasets manually if you want to use them:
- [ImageNet](https://www.image-net.org/)
- [ImageNetR](https://github.com/hendrycks/imagenet-r)
- [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

and prepare them following [Documentation for ImageNetR](/common/vision/datasets/imagenet_r.py) and [ImageNet-Sketch](/common/vision/datasets/imagenet_sketch.py).

## Supported Methods

Supported methods include:

- [Domain Adversarial Neural Network (DANN)](https://arxiv.org/abs/1505.07818)
- [Deep Adaptation Network (DAN)](https://arxiv.org/pdf/1502.02791)
- [Joint Adaptation Network (JAN)](https://arxiv.org/abs/1605.06636)
- [Conditional Domain Adversarial Network (CDAN)](https://arxiv.org/abs/1705.10667)
- [Margin Disparity Discrepancy (MDD)](https://arxiv.org/abs/1904.05801)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/image_classification.rst) with specified hyper-parameters.
For example, if you want to train DANN on Office31, use the following script

```shell script
# Train a DANN on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

## TODO
Support methods: AdaBN/TransNorm, CycleGAN, CyCADA

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{DANN,
    author = {Ganin, Yaroslav and Lempitsky, Victor},
    Booktitle = {ICML},
    Title = {Unsupervised domain adaptation by backpropagation},
    Year = {2015}
}

@inproceedings{DAN,
    author    = {Mingsheng Long and
    Yue Cao and
    Jianmin Wang and
    Michael I. Jordan},
    title     = {Learning Transferable Features with Deep Adaptation Networks},
    booktitle = {ICML},
    year      = {2015},
}

@inproceedings{JAN,
    title={Deep transfer learning with joint adaptation networks},
    author={Long, Mingsheng and Zhu, Han and Wang, Jianmin and Jordan, Michael I},
    booktitle={ICML},
    year={2017},
}

@inproceedings{CDAN,
    author    = {Mingsheng Long and
                Zhangjie Cao and
                Jianmin Wang and
                Michael I. Jordan},
    title     = {Conditional Adversarial Domain Adaptation},
    booktitle = {NeurIPS},
    year      = {2018}
}

@inproceedings{MDD,
    title={Bridging theory and algorithm for domain adaptation},
    author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
    booktitle={ICML},
    year={2019},
}

```
