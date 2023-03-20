# SparK: the first successful BERT/MAE-style pretraining on any convolutional networks &nbsp;[![Reddit](https://img.shields.io/badge/Reddit-🔥%20120k%20views-b31b1b.svg?style=social&logo=reddit)](https://www.reddit.com/r/MachineLearning/comments/10ix0l1/r_iclr2023_spotlight_the_first_bertstyle/) [![Twitter](https://img.shields.io/badge/Twitter-🔥%2020k%2B120k%20views-b31b1b.svg?style=social&logo=twitter)](https://twitter.com/keyutian/status/1616606179144380422)

This is the official implementation of ICLR paper [Designing BERT for Convolutional Networks: ***Spar***se and Hierarchical Mas***k***ed Modeling](https://arxiv.org/abs/2301.03580).
We've tried our best to make the codebase clean, short, easy to read, state-of-the-art, and only rely on minimal dependencies.

<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211496814-e6cb9243-833c-43d2-a859-d35afa96ed22.png" width=86% class="center">
</p>

<div align="center">

  [![SOTA](https://img.shields.io/badge/State%20of%20the%20Art-Self--Supervised%20Image%20Classification%20on%20ImageNet%20%28CNN%29-32B1B4?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgb3ZlcmZsb3c9ImhpZGRlbiI%2BPGRlZnM%2BPGNsaXBQYXRoIGlkPSJjbGlwMCI%2BPHJlY3QgeD0iLTEiIHk9Ii0xIiB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIvPjwvY2xpcFBhdGg%2BPC9kZWZzPjxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMCkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEgMSkiPjxyZWN0IHg9IjUyOSIgeT0iNjYiIHdpZHRoPSI1NiIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIxOSIgeT0iNjYiIHdpZHRoPSI1NyIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIyNzQiIHk9IjE1MSIgd2lkdGg9IjU3IiBoZWlnaHQ9IjMwMiIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjEwNCIgeT0iMTUxIiB3aWR0aD0iNTciIGhlaWdodD0iMzAyIiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNDQ0IiB5PSIxNTEiIHdpZHRoPSI1NyIgaGVpZ2h0PSIzMDIiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIzNTkiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjI2NCIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjE4OCIgeT0iMTcwIiB3aWR0aD0iNTciIGhlaWdodD0iMjY0IiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNzYiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI3NiIgeT0iNDgyIiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjQ4MiIgd2lkdGg9IjQ3IiBoZWlnaHQ9IjU3IiBmaWxsPSIjNDRGMkY2Ii8%2BPC9nPjwvc3ZnPg%3D%3D)](https://paperswithcode.com/sota/self-supervised-image-classification-on-1?tag_filter=17&p=designing-bert-for-convolutional-networks)&nbsp;
  [![OpenReview](https://img.shields.io/badge/ICLR'2023%20Spotlight-NRxydtWup1S-b31b1b.svg)](https://openreview.net/forum?id=NRxydtWup1S)&nbsp;
  [![arXiv](https://img.shields.io/badge/arXiv-2301.03580-b31b1b.svg)](https://arxiv.org/abs/2301.03580)
</div>
  
<!-- <div align="center"> -->
<!--   [[`pdf`](https://arxiv.org/pdf/2301.03580.pdf)] -->
<!--   [[`bibtex`](https://github.com/keyu-tian/SparK#citation)] -->
<!-- </div> -->

## 🔥 News

- The share on [TechBeat (将门创投)](https://www.techbeat.net/talk-info?id=758) is scheduled on **Mar. 16th (UTC+0 12am)** too! [[`📹Recorded Video`](https://www.techbeat.net/talk-info?id=758)]
- We are honored to be invited by Synced ("机器之心机动组 视频号" on WeChat) to give a talk about SparK on **Feb. 27th (UTC+0 11am, UTC+8 7pm)**, welcome! [[`📹Recorded Video`](https://www.bilibili.com/video/BV1J54y1u7U3/)]
- This work got accepted to ICLR 2023 as a Spotlight (notable-top-25%).
- Other articles: [[`Synced`](https://syncedreview.com/2023/01/19/bert-style-pretraining-on-convnets-peking-u-bytedance-oxford-us-sparse-masked-modelling-with-hierarchy-leads-the-way/)]
  [[`DeepAI`](https://deepai.org/publication/designing-bert-for-convolutional-networks-sparse-and-hierarchical-masked-modeling)]
  [[`TheGradient`](https://thegradientpub.substack.com/p/update-42-ai-news-editors-make-mistakes)]
  [[`Bytedance`](https://mp.weixin.qq.com/s/Ak1CeeG83sgO0Wf8KgEIQQ)]
  [[`CVers`](https://zhuanlan.zhihu.com/p/598056871)
  [[`QbitAI(量子位)`](https://www.qbitai.com/2023/02/42109.html)]
  [[`BAAI(智源)`](https://hub.baai.ac.cn/view/23360)]
  [[`机器之心机动组`](https://mp.weixin.qq.com/s/Ylek_lf5enYHRTnkEwAFpg)]
  [[`极市平台`](https://mp.weixin.qq.com/s/GSVHUtBNw5k5wfn2pbC99Q)]
  [[`ReadPaper笔记`](https://readpaper.com/paper/4710371282714116097)]


## 📺 Video demo

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
`reso.`: the image resolution; `acc@1`: IN1k fine-tuned acc (top-1)


|     arch.      | reso. | acc@1 | #params | flops  | weights on google drive                                                                                                                   |
|:--------------:|:-----:|:-----:|:-------:|:------:|:------------------------------------------------------------------------------------------------------------------------------------------|
|    ResNet50    |  224  | 80.6  |   26M   |  4.1G  | [resnet50_1kpretrained_timm_style.pth](https://drive.google.com/file/d/1H8605HbxGvrsu4x4rIoNr-Wkd7JkxFPQ/view?usp=share_link)             |
|   ResNet101    |  224  | 82.2  |   45M   |  7.9G  | [resnet101_1kpretrained_timm_style.pth](https://drive.google.com/file/d/1ZwTztjU-_rfvOVfLoce9SMw2Fx0DQfoO/view?usp=share_link)            |
|   ResNet152    |  224  | 82.7  |   60M   | 11.6G  | [resnet152_1kpretrained_timm_style.pth](https://drive.google.com/file/d/1FOVuECnzQAI-OzE-hnrqW7tVpg8kTziM/view?usp=share_link)            |
|   ResNet200    |  224  | 83.1  |   65M   | 15.1G  | [resnet200_1kpretrained_timm_style.pth](https://drive.google.com/file/d/1_Q4e30qqhjchrdyW3fT6P98Ga-WnQ57s/view?usp=share_link)            |
|   ConvNeXt-S   |  224  | 84.1  |   50M   |  8.7G  | [convnextS_1kpretrained_official_style.pth](https://drive.google.com/file/d/1Ah6lgDY5YDNXoXHQHklKKMbEd08RYivN/view?usp=share_link)        |
|   ConvNeXt-B   |  224  | 84.8  |   89M   | 15.4G  | [convnextB_1kpretrained_official_style.pth](https://drive.google.com/file/d/1ZjWbqI1qoBcqeQijI5xX9E-YNkxpJcYV/view?usp=share_link)        |
|   ConvNeXt-L   |  224  | 85.4  |  198M   | 34.4G  | [convnextL_1kpretrained_official_style.pth](https://drive.google.com/file/d/1qfYzGUpYBzuA88_kXkVl4KNUwfutMVfw/view?usp=share_link)        |
|   ConvNeXt-L   |  384  | 86.0  |  198M   | 101.0G | [convnextL_384_1kpretrained_official_style.pth](https://drive.google.com/file/d/1YgWNXJjI89l35P4ksAmBNWZ2JZCpj9n4/view?usp=share_link)    |
| L-with-decoder |  384  | 86.0  |  198M   | 101.0G | [cnxL384_withdecoder_1kpretrained_spark_style.pth](https://drive.google.com/file/d/1ZI9Jgtb3fKWE_vDFEly29w-1FWZSNwa0/view?usp=share_link) |



## Installation

For pre-training and fine-tuning on ImageNet-1k, we highly recommended you to use `torch==1.10.0`, `torchvision==0.11.1`, and `timm==0.5.4`.

Check [INSTALL.md](INSTALL.md) to install all dependencies for pre-training and ImageNet fine-tuning.

## Pre-training

See [pretrain/](pretrain) to pre-train models on ImageNet-1k.


## Fine-tuning

- All models on ImageNet: check [downstream_imagenet/](downstream_imagenet) for subsequent instructions.
- ResNets on COCO: see [downstream_d2/](downstream_d2) for details.
- ConvNeXts on COCO: see [downstream_mmdet/](downstream_mmdet) for details.


## Acknowledgement

We referred to these useful codebases:

- [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [MAE](https://github.com/facebookresearch/mae), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [timm](https://github.com/rwightman/pytorch-image-models), [MoCoV2](https://github.com/facebookresearch/moco), [Detectron2](https://github.com/facebookresearch/detectron2), [MMDetection](https://github.com/open-mmlab/mmdetection)



## License
This project is under the MIT license. See [LICENSE](LICENSE) for more details.


## Citation

If you found this project useful, you can kindly give us a star ⭐, or cite us in your work 📖:
```
@Article{tian2023designing,
  author  = {Keyu Tian and Yi Jiang and Qishuai Diao and Chen Lin and Liwei Wang and Zehuan Yuan},
  title   = {Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling},
  journal = {arXiv:2301.03580},
  year    = {2023},
}
```

