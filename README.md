## The Official PyTorch Implementation of SparK🔥 (Sparse and Hierarchical Masked Modeling) [![arXiv](https://img.shields.io/badge/arXiv-2301.03580-b31b1b.svg)](https://arxiv.org/abs/2301.03580)

<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211496814-e6cb9243-833c-43d2-a859-d35afa96ed22.png" width=86% class="center">
</p>

<div align="center">

  [[`arXiv`](https://arxiv.org/abs/2301.03580)]
  [[`pdf`](https://www.researchgate.net/profile/Keyu-Tian-2/publication/366984303_Designing_BERT_for_Convolutional_Networks_Sparse_and_Hierarchical_Masked_Modeling/links/63bcf24bc3c99660ebe253c5/Designing-BERT-for-Convolutional-Networks-Sparse-and-Hierarchical-Masked-Modeling.pdf)]
  [[`state-of-the-art self-supervised convnet`](https://paperswithcode.com/sota/self-supervised-image-classification-on-1?tag_filter=17?p=designing-bert-for-convolutional-networks)]
  [[`bibtex`](https://github.com/keyu-tian/SparK#citation)]
</div>


## Introduction

This is an official implementation of the paper: "Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling".
We'll be updating frequently these days, so you might consider star ⭐ or watch 👓 this repository to get the latest information.

In this work we designed a BERT-style pre-training framework (a.k.a. masked image modeling) for any hierarchical (multi-scale) convnets.
As shown above, it gathers all unmasked patches to form a sparse image and uses sparse convolution for encoding.
A dense, hierarchical decoder is applied then, to reconstruct all masked pixels.
This method is general and powerful: it can be used directly on any convolutional backbones such as classical ResNets (the right) and modern ConvNeXts (left), and can bring a leap in their performance:


<img src="https://user-images.githubusercontent.com/39692511/211497396-cd031318-ef54-45a4-a283-cd9810c15603.png" width=45%><img src="https://user-images.githubusercontent.com/39692511/211497479-0563e891-f2ad-4cf1-b682-a21c2be1442d.png" width=55%>


See our [paper](https://www.researchgate.net/profile/Keyu-Tian-2/publication/366984303_Designing_BERT_for_Convolutional_Networks_Sparse_and_Hierarchical_Masked_Modeling/links/63bcf24bc3c99660ebe253c5/Designing-BERT-for-Convolutional-Networks-Sparse-and-Hierarchical-Masked-Modeling.pdf) for more analysis, discussion, and evaluation.


## Pre-train

See [PRETRAIN.md](PRETRAIN.md) for preparation and pre-training.

## Fine-tune on ImageNet

After finishing the preparation in [PRETRAIN.md](PRETRAIN.md), see [downstream_imagenet](downstream_imagenet) for subsequent instructions.

## Fine-tune ResNets on COCO

Install `Detectron2` and see [downstream_d2](downstream_d2) for more details.

## Fine-tune ConvNeXts on COCO

Install `mmcv` and `mmdetection` then see [downstream_mmdet](downstream_mmdet) for more details.


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
This project is under the CC-BY 4.0 license. See [LICENSE](LICENSE) for more details.


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

