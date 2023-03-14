# SparK✨: The first successful BERT-style pre-training on any convolutional networks [![arXiv](https://img.shields.io/badge/arXiv-2301.03580-b31b1b.svg)](https://arxiv.org/abs/2301.03580), ICLR'23 Spotlight

Official implementation of "Designing BERT for Convolutional Networks: ***Spar***se and Hierarchical Mas***k***ed Modeling".


<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211496814-e6cb9243-833c-43d2-a859-d35afa96ed22.png" width=86% class="center">
</p>

<div align="center">

  [[`arXiv`](https://arxiv.org/abs/2301.03580)]
  [[`pdf`](https://arxiv.org/pdf/2301.03580.pdf)]
  [[`state-of-the-art self-supervised convnet`](https://paperswithcode.com/sota/self-supervised-image-classification-on-1?tag_filter=17?p=designing-bert-for-convolutional-networks)]
  [[`bibtex`](https://github.com/keyu-tian/SparK#citation)]
</div>

<div align="center">

  [[`ReadPaper`](https://readpaper.com/paper/4710371282714116097)]
  [[`Synced`](https://syncedreview.com/2023/01/19/bert-style-pretraining-on-convnets-peking-u-bytedance-oxford-us-sparse-masked-modelling-with-hierarchy-leads-the-way/)]
  [[`The Gradient`](https://thegradientpub.substack.com/p/update-42-ai-news-editors-make-mistakes)]
  [[`QbitAI`](https://www.qbitai.com/2023/02/42109.html)]
  [[`Bytedance`](https://mp.weixin.qq.com/s/Ak1CeeG83sgO0Wf8KgEIQQ)]
  [[`DeepAI`](https://deepai.org/publication/designing-bert-for-convolutional-networks-sparse-and-hierarchical-masked-modeling)]
  [[`Reddit`](https://www.reddit.com/r/MachineLearning/comments/10ix0l1/r_iclr2023_spotlight_the_first_bertstyle/)]
  [[`Twitter`](https://twitter.com/keyutian/status/1616606179144380422)]
</div>

## 🔥 News

- We are honored to be invited by Synced ("机器之心机动组 视频号" on WeChat) to give a talk about SparK on **Feb. 27th (UTC+0 11am, UTC+8 7pm)**, welcome! [[`Recorded Video`](https://www.bilibili.com/video/BV1J54y1u7U3/)]
- Another share on [TechBeat (将门创投)](https://www.techbeat.net/talk-info?id=758) is scheduled on **Mar. 16th (UTC+0 12am, UTC+8 8pm)** too! [[`Recorded Video`](https://www.techbeat.net/talk-info?id=758)]


## Video demo

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
- [x] Fine-tuning code
- [ ] Colab visualization playground
- [ ] Weights & visualization playground on `Huggingface`
- [ ] Weights in `timm`


## ImageNet-1k results and pre-trained networks weights

**Note: for network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**

| arch. | acc@1 | #params | flops | model |
|:---:|:---:|:---:|:---:|:---:|
| ResNet50   | 80.6 | 26M  | 4.1G | [drive](https://drive.google.com/file/d/1H8605HbxGvrsu4x4rIoNr-Wkd7JkxFPQ/view?usp=share_link) |
| ResNet101  | 82.2 | 45M  | 7.9G | [drive](https://drive.google.com/file/d/1ZwTztjU-_rfvOVfLoce9SMw2Fx0DQfoO/view?usp=share_link) |
| ResNet152  | 82.7 | 60M | 11.6G | [drive](https://drive.google.com/file/d/1FOVuECnzQAI-OzE-hnrqW7tVpg8kTziM/view?usp=share_link) |
| ResNet200  | 83.1 | 65M | 15.1G | [drive](https://drive.google.com/file/d/1_Q4e30qqhjchrdyW3fT6P98Ga-WnQ57s/view?usp=share_link) |
| ConvNeXt-S | 84.1 | 50M  | 8.7G | [drive](https://drive.google.com/file/d/1Ah6lgDY5YDNXoXHQHklKKMbEd08RYivN/view?usp=share_link) |
| ConvNeXt-B | 84.8 | 89M  | 15.4G | [drive](https://drive.google.com/file/d/1ZjWbqI1qoBcqeQijI5xX9E-YNkxpJcYV/view?usp=share_link) |
| ConvNeXt-L | 85.4 | 198M | 34.4G | [drive](https://drive.google.com/file/d/1qfYzGUpYBzuA88_kXkVl4KNUwfutMVfw/view?usp=share_link) |



## Installation

For pre-training and fine-tuning on ImageNet-1k, we highly recommended you to use `torch==1.10.0`, `torchvision==0.11.1`, and `timm==0.5.4`.

Check [INSTALL.md](INSTALL.md) to install all dependencies for pre-training and ImageNet fine-tuning.

## Pre-training

See [PRETRAIN.md](PRETRAIN.md) to pre-train models on ImageNet-1k.


## Fine-tuning

- Models on ImageNet: after installation, check [downstream_imagenet](downstream_imagenet) for subsequent instructions.
- ResNets on COCO: install `detectron2` and see [downstream_d2](downstream_d2) for more details.
- ConvNeXts on COCO: install `mmcv` and `mmdetection` then see [downstream_mmdet](downstream_mmdet) for more details.


## Acknowledgement

We referred to these useful codebases:

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

