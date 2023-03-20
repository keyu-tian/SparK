## Preparation for ImageNet-1k pre-training

See [INSTALL.md](https://github.com/keyu-tian/SparK/blob/main/INSTALL.md) to prepare `pip` dependencies and the ImageNet dataset.

**Note: for network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**


## Pre-training Any Model on ImageNet-1k (224x224)

For pre-training, run [main.sh](https://github.com/keyu-tian/SparK/blob/main/pretrain/main.sh) with bash.
It is **required** to specify the ImageNet data folder (`--data_path`), the model name (`--model`), and your experiment name (the first argument of `main.sh`) when running the script.

We use the **same** pre-training configurations (lr, batch size, etc.) for all models (ResNets and ConvNeXts).
Their names and **default values** can be found in [utils/arg_util.py line24-47](https://github.com/keyu-tian/SparK/blob/main/pretrain/utils/arg_util.py#L24).
These default configurations (like batch size 4096) would be used, unless you specify some like `--bs=512`.

Here is an example command pre-training a ResNet50 on single machine with 8 GPUs:
```shell script
$ cd /path/to/SparK/pretrain
$ bash ./main.sh <experiment_name> \
  --num_nodes=1 --ngpu_per_node=8 \
  --data_path=/path/to/imagenet \
  --model=resnet50 --bs=512
```

For multiple machines, change the `--num_nodes` to your count, and plus these args:
```shell script
--node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port>
```

Note the `<experiment_name>` is the name of your experiment, which would be used to create an output directory named `output_<experiment_name>`.


## Pre-training ConvNeXt-Large on ImageNet-1k (384x384)

For pre-training with resolution 384, we use a larger mask ratio (0.75), a smaller batch size (2048), and a larger learning rate (4e-4):

```shell script
$ cd /path/to/SparK/pretrain
$ bash ./main.sh <experiment_name> \
--num_nodes=8 --ngpu_per_node=8 --node_rank=... --master_address=... --master_port=... \
--data_path=/path/to/imagenet \
--model=convnext_large --input_size=384 --mask=0.75 \
 --bs=2048 --base_lr=4e-4
```

## Logging

Once an experiment starts running, the following files would be automatically created and updated in `output_<experiment_name>`:

- `<model>_still_pretraining.pth`: saves model and optimizer states, current epoch, current reconstruction loss, etc; can be used to resume pre-training
- `<model>_1kpretrained.pth`: can be used for downstream fine-tuning
- `pretrain_log.txt`: records some important information such as:
    - `git_commit_id`: git version
    - `cmd`: all arguments passed to the script
    
    It also reports the loss and remaining pre-training time at each epoch.

- `stdout_backup.txt` and `stderr_backup.txt`: will save all output to stdout/stderr

These files can help trace the experiment well.


## Resuming

Add `--resume_from=path/to/<model>still_pretraining.pth` to resume from a saved checkpoint.


## Regarding sparse convolution

We do not use sparse convolutions in this pytorch implementation, due to their limited optimization on modern hardwares.
As can be found in [encoder.py](https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py), we use masked dense convolution to simulate submanifold sparse convolution.
We also define some sparse pooling or normalization layers in [encoder.py](https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py).
All these "sparse" layers are implemented through pytorch built-in operators.


## Some details: how we mask images and how to set the patch size

In SparK, the mask patch size **equals to** the downsample ratio of the CNN model (so there is no configuration like `--patch_size=32`).

Here is the reason: when we do mask, we:

1. first generate the binary mask for the **smallest** resolution feature map, i.e., generate the `_cur_active` or `active_b1ff` in [line86-87](https://github.com/keyu-tian/SparK/blob/main/pretrain/spark.py#L86), which is a `torch.BoolTensor` shaped as `[B, 1, fmap_size, fmap_size]`, and would be used to mask the smallest feature map.
3. then progressively upsample it (i.e., expand its 2nd and 3rd dimensions by calling `repeat_interleave(..., 2)` and `repeat_interleave(..., 3)` in [line16](https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py#L16)), to mask those feature maps ([`x` in line21](https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py#L21)) with larger resolutions .

So if you want a patch size of 16 or 8, you should actually define a new CNN model with a downsample ratio of 16 or 8.
Note that the `forward` function of this CNN should have an arg named `hierarchy`. You can look at https://github.com/keyu-tian/SparK/blob/main/pretrain/models/convnext.py#L78 to see what `hierarchy` means and how to handle it.

After that, you can simply run `main.sh` with `--hierarchy=3` and see if it works.
