# SparK✨: the first successful BERT-style pre-training on any convolutional networks [![arXiv](https://img.shields.io/badge/arXiv-2301.03580-b31b1b.svg)](https://arxiv.org/abs/2301.03580)

This is an official implementation of the paper "Designing BERT for Convolutional Networks: ***Spar***se and Hierarchical Mas***k***ed Modeling". (submitted to [openreview ICLR'23](https://openreview.net/forum?id=NRxydtWup1S) in Sep. 2022)


<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211496814-e6cb9243-833c-43d2-a859-d35afa96ed22.png" width=86% class="center">
</p>

<div align="center">

  [[`arXiv`](https://arxiv.org/abs/2301.03580)]
  [[`pdf`](https://arxiv.org/pdf/2301.03580.pdf)]
  [[`state-of-the-art self-supervised convnet`](https://paperswithcode.com/sota/self-supervised-image-classification-on-1?tag_filter=17?p=designing-bert-for-convolutional-networks)]
  [[`bibtex`](https://github.com/keyu-tian/SparK#citation)]
</div>

## A quick explainable video demo for spark

https://user-images.githubusercontent.com/6366788/213662770-5f814de0-cbe8-48d9-8235-e8907fd81e0e.mp4

## What's new here?

### 🔥 On ResNets, generative pre-training surpasses contrastive learning for the first time:

<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211497479-0563e891-f2ad-4cf1-b682-a21c2be1442d.png" width=68%>
<p>

### 🔥 ConvNeXt gains more from pre-training than Swin-Transformer, up to +3.5 points:

<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211497396-cd031318-ef54-45a4-a283-cd9810c15603.png" width=68%>
<p>

### 🔥 Larger models benefit more from SparK pre-training, showing a scaling behavior:


<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211705760-de15f4a1-0508-4690-981e-5640f4516d2a.png" width=68%>
<p>


### 🔥 Pre-trained model can make reasonable predictions:

<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211703443-220495d5-452a-446d-b7c7-c66a0c19741a.png" width=85%>
<p>

#### See our [paper](https://arxiv.org/pdf/2301.03580.pdf) for more analysis, discussions, and evaluations.


## Catalog

- [x] Pre-training code
- [ ] Fine-tuning code
- [ ] Colab playground
- [ ] Inference and visualization demo

## Installation

Check [INSTALL.md](INSTALL.md) to install all dependencies. Our implementation is based on `torch==1.10.0+cu113`, `torchvision==0.11.1+cu113`, and `timm==0.5.4`. [This](https://github.com/facebookresearch/SparseConvNet) sparse convolution framework is an optional library.

## Pre-training

See [PRETRAIN.md](PRETRAIN.md) to pre-train models on ImageNet.

## Fine-tuning

- Models on ImageNet: after installation, check [downstream_imagenet](downstream_imagenet) for subsequent instructions.
- ResNets on COCO: install `detectron2` and see [downstream_d2](downstream_d2) for more details.
- ConvNeXts on COCO: install `mmcv` and `mmdetection` then see [downstream_mmdet](downstream_mmdet) for more details.


## Acknowledgement


We heavily referred to these useful codebases:

- [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
- [MAE](https://github.com/facebookresearch/mae)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

We also appreciate these elegant frameworks:

- [timm](https://github.com/rwightman/pytorch-image-models)
- [MoCoV2](https://github.com/facebookresearch/moco)
- [Detectron2](https://github.com/facebookresearch/detectron2) and [MMDetection](https://github.com/open-mmlab/mmdetection)



## License
This project is under the MIT license. See [LICENSE](LICENSE) for more details.


## Citation

If you found this project useful, please consider adding a star ⭐, or citing us 📖:
```
@Article{tian2023designing,
  author  = {Keyu Tian and Yi Jiang and Qishuai Diao and Chen Lin and Liwei Wang and Zehuan Yuan},
  title   = {Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling},
  journal = {arXiv:2301.03580},
  year    = {2023},
}
```

