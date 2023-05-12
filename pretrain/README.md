## Preparation for ImageNet-1k pretraining

See [/INSTALL.md](/INSTALL.md) to prepare `pip` dependencies and the ImageNet dataset.

**Note: for network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**


## Tutorial for pretraining your own CNN model

See [/pretrain/models/custom.py](/pretrain/models/custom.py). The things needed to do is:

- implementing member function `get_downsample_ratio` in [/pretrain/models/custom.py line20](/pretrain/models/custom.py#L20).
- implementing member function `get_feature_map_channels` in [/pretrain/models/custom.py line29](/pretrain/models/custom.py#L29).
- implementing member function `forward` in [/pretrain/models/custom.py line38](/pretrain/models/custom.py#L38).
- define `your_convnet(...)` with `@register_model` in [/pretrain/models/custom.py line54](/pretrain/models/custom.py#L53-L54).
- add default kwargs of `your_convnet(...)` in [/pretrain/models/\_\_init\_\_.py line34](/pretrain/models/__init__.py#L34).

Then you can use `--model=your_convnet` in the pretraining script.


## Tutorial for pretraining your own dataset

Replace the function `build_dataset_to_pretrain` in [line54-75 of /pretrain/utils/imagenet.py](/pretrain/utils/imagenet.py#L54-L75) to yours.
This function should return a `Dataset` object. You may use args like `args.data_path` and `args.input_size` to help build your dataset. And when runing experiment with `main.sh` you can use `--data_path=... --input_size=...` to specify them.
Note the batch size `--bs` is the total batch size of all GPU, which may also need to be tuned.


## Debug on 1 GPU (without DistributedDataParallel)

Use a small batch size `--bs=32` for avoiding OOM.

```shell script
python3 main.py --exp_name=debug --data_path=/path/to/imagenet --model=resnet50 --bs=32
```


## Pretraining Any Model on ImageNet-1k (224x224)

For pretraining, run [/pretrain/main.sh](/pretrain/main.sh) with bash.
It is **required** to specify the ImageNet data folder (`--data_path`), the model name (`--model`), and your experiment name (the first argument of `main.sh`) when running the script.

We use the **same** pretraining configurations (lr, batch size, etc.) for all models (ResNets and ConvNeXts).
Their names and **default values** can be found in [/pretrain/utils/arg_util.py line23-44](/pretrain/utils/arg_util.py#L23-L44).
These default configurations (like batch size 4096) would be used, unless you specify some like `--bs=512`.

**Note: the batch size `--bs` is the total batch size of all GPU, and the learning rate `--base_lr` is the base learning rate. The actual learning rate would be `lr * bs / 256`, as in [/pretrain/utils/arg_util.py line131](/pretrain/utils/arg_util.py#L131). Don't use `--lr` to specify a lr (would be ignored)**

Here is an example command pretraining a ResNet50 on single machine with 8 GPUs (we use DistributedDataParallel):
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


## Pretraining ConvNeXt-Large on ImageNet-1k (384x384)

For pretraining with resolution 384, we use a larger mask ratio (0.75), a smaller batch size (2048), and a larger learning rate (4e-4):

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

- `<model>_still_pretraining.pth`: saves model and optimizer states, current epoch, current reconstruction loss, etc; can be used to resume pretraining
- `<model>_1kpretrained.pth`: can be used for downstream finetuning
- `pretrain_log.txt`: records some important information such as:
    - `git_commit_id`: git version
    - `cmd`: all arguments passed to the script
    
    It also reports the loss and remaining pretraining time at each epoch.

- `stdout_backup.txt` and `stderr_backup.txt`: will save all output to stdout/stderr

These files can help trace the experiment well.


## Resuming

Add `--resume_from=path/to/<model>still_pretraining.pth` to resume from a saved checkpoint.


## Regarding sparse convolution

We do not use sparse convolutions in this pytorch implementation, due to their limited optimization on modern hardwares.
As can be found in [/pretrain/encoder.py](/pretrain/encoder.py), we use masked dense convolution to simulate submanifold sparse convolution.
We also define some sparse pooling or normalization layers in [/pretrain/encoder.py](/pretrain/encoder.py).
All these "sparse" layers are implemented through pytorch built-in operators.


## Some details: how we mask images and how to set the patch size

In SparK, the mask patch size **equals to** the downsample ratio of the CNN model (so there is no configuration like `--patch_size=32`).

Here is the reason: when we do mask, we:

1. first generate the binary mask for the **smallest** resolution feature map, i.e., generate the `_cur_active` or `active_b1ff` in [/pretrain/spark.py line86-87](/pretrain/spark.py#L86-L87), which is a `torch.BoolTensor` shaped as `[B, 1, fmap_size, fmap_size]`, and would be used to mask the smallest feature map.
3. then progressively upsample it (i.e., expand its 2nd and 3rd dimensions by calling `repeat_interleave(..., 2)` and `repeat_interleave(..., 3)` in [/pretrain/encoder.py line16](/pretrain/encoder.py#L16)), to mask those feature maps ([`x` in line21](/pretrain/encoder.py#L21)) with larger resolutions .

So if you want a patch size of 16 or 8, you should actually define a new CNN model with a downsample ratio of 16 or 8.
See [Tutorial for pretraining your own CNN model (above)](https://github.com/keyu-tian/SparK/tree/main/pretrain/#tutorial-for-pretraining-your-own-cnn-model).
