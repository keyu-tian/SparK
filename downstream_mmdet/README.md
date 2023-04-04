## About code isolation

This `downstream_mmdet` is isolated from pre-training codes. One can treat this `downstream_mmdet` as an independent codebase 🛠️.

## Fine-tuned ConvNeXt-B weights, log files, and performance


<div align="center">

[[`weights (pre-trained by SparK)`](https://drive.google.com/file/d/1ZjWbqI1qoBcqeQijI5xX9E-YNkxpJcYV/view?usp=share_link)]
  [[`weights (fine-tuned on COCO)`](https://drive.google.com/file/d/1t10dmzg5KOO27o2yIglK-gQepB5gR4zR/view?usp=share_link)]
  [[`log.json`](https://drive.google.com/file/d/1TuNboXl1qwjf1tggZ3QOssI67uU7Jtig/view?usp=share_link)]
  [[`log`](https://drive.google.com/file/d/1JY5CkL_MX08zJ8P1FBIeC60OJsuIiyZc/view?usp=sharing)]
</div>


<p align="center">
<img src="https://user-images.githubusercontent.com/39692511/211497396-cd031318-ef54-45a4-a283-cd9810c15603.png" width=80%>
<p>


## Installation [MMDetection with commit 6a979e2](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/6a979e2164e3fb0de0ca2546545013a4d71b2f7d) before fine-tuning ConvNeXt on COCO

We refer to the codebases of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/048efcea897d999aed302f2639b6270aedf8d4c8) and [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/6a979e2164e3fb0de0ca2546545013a4d71b2f7d).
Please refer to [README.md](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/6a979e2164e3fb0de0ca2546545013a4d71b2f7d/README.md) for installation and dataset preparation instructions.

Note the COCO dataset folder should be at `downstream_mmdet/data/coco`.
   The folder should follow the directory structure requried by `MMDetection`, which should look like this:
```
downstream_mmdet/data/coco:
    annotations/:
        captions_train2017.json  captions_val2017.json
        instances_train2017.json  instances_val2017.json
        person_keypoints_train2017.json  person_keypoints_val2017.json
    train2017/:
        a_lot_images.jpg
    val2017/:
        a_lot_images.jpg
```


### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [other optional arguments] 
```
For example, to train a Mask R-CNN model with a SparK pretrained `ConvNeXt-B` backbone and 4 gpus, run:
```
tools/dist_train.sh configs/convnext_spark/mask_rcnn_convnext_base_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py 4 \
  --cfg-options model.pretrained=/some/path/to/official_convnext_base_1kpretrained.pth
```

The Mask R-CNN 3x fine-tuning config file can be found at [`configs/convnext_spark`](configs/convnext_spark). This config is basically a copy of [https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/configs/convnext/mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py](https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/configs/convnext/mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py).

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

## Acknowledgment 

We appreciate these useful codebases:

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)

