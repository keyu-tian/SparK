## Preparation for ImageNet-1k pretraining

See [/INSTALL.md](/INSTALL.md) to prepare `pip` dependencies and the ImageNet dataset.

**Note: for neural network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**


## Tutorial for pretraining your own CNN model

See [/pretrain/models/custom.py](/pretrain/models/custom.py). Your todo list is:

- implement `get_downsample_ratio` in [/pretrain/models/custom.py line20](/pretrain/models/custom.py#L20).
- implement `get_feature_map_channels` in [/pretrain/models/custom.py line29](/pretrain/models/custom.py#L29).
- implement `forward` in [/pretrain/models/custom.py line38](/pretrain/models/custom.py#L38).
- define `your_convnet(...)` with `@register_model` in [/pretrain/models/custom.py line54](/pretrain/models/custom.py#L53-L54).
- add default kwargs of `your_convnet(...)` in [/pretrain/models/\_\_init\_\_.py line34](/pretrain/models/__init__.py#L34).
- **Note: see [#54](/../../issues/54) if your CNN contains SE module or global average pooling layer, and see [#56](/../../issues/56) if it contains GroupNorm**.

Then run the experiment with `--model=your_convnet`.


## Tutorial for pretraining on your own dataset

See the comment of `build_dataset_to_pretrain` in [line55 of /pretrain/utils/imagenet.py](/pretrain/utils/imagenet.py#L55). Your todo list:

- Define a subclass of `torch.utils.data.Dataset` for your own unlabeled dataset, to replace our `ImageNetDataset`.
- Use `args.data_path` and `args.input_size` to help build your dataset, with `--data_path=... --input_size=...` to specify them.
- Note the batch size `--bs` is the total batch size of all GPU, which may need to be adjusted based on your dataset size. FYI: we use `--bs=4096` for ImageNet, which contains 1.28 million images.

**If your dataset is relatively small**, you can try `--init_weight=/path/to/res50_withdecoder_1kpretrained_spark_style.pth` to do your pretraining *from our pretrained weights*, rather than *form scratch*.

## Debug on 1 GPU (without DistributedDataParallel)

Use a small batch size `--bs=32` for avoiding OOM.

```shell script
python3 main.py --exp_name=debug --data_path=/path/to/imagenet --model=resnet50 --bs=32
```


## Pretraining Any Model on ImageNet-1k (224x224)

For pretraining, run [/pretrain/main.py](/pretrain/main.py) with `torchrun`.
**It is required to specify** the ImageNet data folder (`--data_path`), your experiment name & log dir (`--exp_name` and `--exp_dir`, automatically created if not exists), and the model name (`--model`, valid choices see the keys of 'pretrain_default_model_kwargs' in [/pretrain/models/\_\_init\_\_.py line34](/pretrain/models/__init__.py#L34)).

We use the **same** pretraining configurations (lr, batch size, etc.) for all models (ResNets and ConvNeXts) in 224 pretraining.
Their **names** and **default values** are in [/pretrain/utils/arg_util.py line23-44](/pretrain/utils/arg_util.py#L23-L44).
All these default configurations (like batch size 4096) would be used, unless you specify some like `--bs=512`.

**Note: the batch size `--bs` is the total batch size of all GPU, and the learning rate `--base_lr` is the base lr. The actual lr would be `lr = base_lr * bs / 256`, as in [/pretrain/utils/arg_util.py line131](/pretrain/utils/arg_util.py#L131). So do not use `--lr` to specify a lr (that will be ignored)**

Here is an example to pretrain a ResNet50 on an 8-GPU single machine (we use DistributedDataParallel), overwriting the default batch size to 512:
```shell script
$ cd /path/to/SparK/pretrain
$ torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=<some_port> main.py \
  --data_path=/path/to/imagenet --exp_name=<your_exp_name> --exp_dir=/path/to/logdir \
  --model=resnet50 --bs=512
```

For multiple machines, change the `--nnodes`, `--node_rank`, `--master_address` and `--master_port` to your configurations. E.g.:
```shell script
$ torchrun --nproc_per_node=8 --nnodes=<your_nnodes> --node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port> main.py \
  ...
```

## Pretraining ConvNeXt-Large on ImageNet-1k (384x384)

For 384 pretraining we use a larger mask ratio (0.75), a half batch size (2048), and a double base learning rate (4e-4):

```shell script
$ cd /path/to/SparK/pretrain
$ torchrun --nproc_per_node=8 --nnodes=<your_nnodes> --node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port> main.py \
  --data_path=/path/to/imagenet --exp_name=<your_exp_name> --exp_dir=/path/to/logdir \
  --model=convnext_large --input_size=384 --mask=0.75 --bs=2048 --base_lr=4e-4
```

## Logging

See files in your `--exp_dir` to track your experiment:

- `<model>_withdecoder_1kpretrained_spark_style.pth`: saves model and optimizer states, current epoch, current reconstruction loss, etc.; can be used to resume pretraining; can also be used for visualization in [/pretrain/viz_reconstruction.ipynb](/pretrain/viz_reconstruction.ipynb)
- `<model>_1kpretrained_timm_style.pth`: can be used for downstream finetuning
- `pretrain_log.txt`: records some important information such as:
    - `git_commit_id`: git version
    - `cmd`: the command of this experiment
    
    It also reports the loss and remaining pretraining time.

- `tensorboard_log/`: saves a lot of tensorboard logs including loss values, learning rates, gradient norms and more things. Use `tensorboard --logdir /path/to/this/tensorboard_log/ --port 23333` for viz.
- `stdout_backup.txt` and `stderr_backup.txt`: backups stdout/stderr.

## Resuming

Specify `--resume_from=path/to/<model>_withdecoder_1kpretrained_spark_style.pth` to resume pretraining. Note this is different from `--init_weight`:

- `--resume_from` will load three things: model weights, optimizer states, and current epoch, so it is used to resume some interrupted experiment (will start from that 'current epoch').
- `--init_weight` ONLY loads the model weights, so it's just like a model initialization (will start from epoch 0).


## Regarding sparse convolution

We do not use sparse convolutions in this pytorch implementation, due to their limited optimization on modern hardware.
As can be found in [/pretrain/encoder.py](/pretrain/encoder.py), we use masked dense convolution to simulate submanifold sparse convolution.
We also define some sparse pooling or normalization layers in [/pretrain/encoder.py](/pretrain/encoder.py).
All these "sparse" layers are implemented through pytorch built-in operators.


## Some details: how we mask images and how to set the patch size

In SparK, the mask patch size **equals to** the downsample ratio of the CNN model (so there is no configuration like `--patch_size=32`).

Here is the reason: when we do mask, we:

1. first generate the binary mask for the **smallest** resolution feature map, i.e., generate the `_cur_active` or `active_b1ff` in [/pretrain/spark.py line86-87](/pretrain/spark.py#L86-L87), which is a `torch.BoolTensor` shaped as `[B, 1, fmap_h, fmap_w]`, and would be used to mask the smallest feature map.
3. then progressively upsample it (i.e., expand its 2nd and 3rd dimensions by calling `repeat_interleave(..., dim=2)` and `repeat_interleave(..., dim=3)` in [/pretrain/encoder.py line16](/pretrain/encoder.py#L16)), to mask those feature maps ([`x` in line21](/pretrain/encoder.py#L21)) with larger resolutions .

So if you want a patch size of 16 or 8, you should actually define a new CNN model with a downsample ratio of 16 or 8.
See [Tutorial for pretraining your own CNN model (above)](https://github.com/keyu-tian/SparK/tree/main/pretrain/#tutorial-for-pretraining-your-own-cnn-model).
