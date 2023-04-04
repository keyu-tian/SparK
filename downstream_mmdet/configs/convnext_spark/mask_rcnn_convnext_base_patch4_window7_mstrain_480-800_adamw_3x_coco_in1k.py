"""
We directly take the ConvNeXt-T+MaskRCNN 3x recipe from https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/configs/convnext/mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py
And we modify  this  ConvNeXt-T+MaskRCNN 3x recipe to our ConvNeXt-B+MaskRCNN 3x recipe.
The modifications (commented as [modified] below) are according to:
- 1. tiny-to-base: (some configs of ConvNext-T are updated to those of ConvNext-B, referring to https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/configs/convnext/cascade_mask_rcnn_convnext_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in22k.py)
    - model.backbone.{depths, dims, drop_path_rate}
    - models.neck
    - optimizer.paramwise_cfg.num_layers

- 2. our paper (https://openreview.net/forum?id=NRxydtWup1S, or https://arxiv.org/abs/2301.03580):
    - LR layer decay (optimizer.paramwise_cfg.decay_rate): 0.65
    - LR scheduled ratio (lr_config.gamma): 0.2
    - Learning rate (optimizer.lr): 0.0002
    - optimizer_config.use_fp16: False (we just use fp32 by default; actually we didn't test the performance of using fp16)
"""

_base_ = [
    '../_base_/models/mask_rcnn_convnext_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        depths=[3, 3, 27, 3],           # [modified] according to tiny-to-base
        dims=[128, 256, 512, 1024],     # [modified] according to tiny-to-base
        drop_path_rate=0.5,             # [modified] according to tiny-to-base
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]))   # [modified] according to tiny-to-base

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,  # [modified] according to our paper
                 paramwise_cfg={'decay_rate': 0.65,                 # [modified] according to our paper
                                'decay_type': 'layer_wise',
                                'num_layers': 12})                  # [modified] according to tiny-to-base
lr_config = dict(step=[27, 33], gamma=0.2)                          # [modified] according to our paper
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,              # [modified] True => False
)