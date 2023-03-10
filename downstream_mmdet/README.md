## About code isolation

This `downstream_mmdet` is isolated from pre-training codes. One can treat this `downstream_mmdet` as an independent codebase 🛠️.

## Getting started 

## Installation [MMDetection with commit 6a979e2](https://github.com/facebookresearch/detectron2/releases/tag/v0.6) before fine-tuning ConvNeXt on COCO

We refer to the codebases of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/048efcea897d999aed302f2639b6270aedf8d4c8) and [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/6a979e2164e3fb0de0ca2546545013a4d71b2f7d).
Please refer to [README.md](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/6a979e2164e3fb0de0ca2546545013a4d71b2f7d/README.md) for installation and dataset preparation instructions.

Note the COCO dataset folder should be at `downstream_mmdet/data/coco`.
   The folder should follow the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets) requried by `Detectron2`, which should look like this:
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
For example, to train a Cascade Mask R-CNN model with a `ConvNeXt-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/convnext/cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py 8 --cfg-options model.pretrained=https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth
```

More config files can be found at [`configs/convnext_spark`](configs/convnext_spark).

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
