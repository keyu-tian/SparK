## About code isolation

This `downstream_imagenet` is isolated from pre-training codes. One can treat this `downstream_imagenet` as an independent codebase 🛠️.


## Preparation for ImageNet-1k fine-tuning

See [INSTALL.md](https://github.com/keyu-tian/SparK/blob/main/INSTALL.md) to prepare `pip` dependencies and the ImageNet dataset.

**Note: for network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**


## Fine-tuning on ImageNet-1k from pre-trained weights

Run [/downstream_imagenet/main.py](/downstream_imagenet/main.py) via `torchrun`.
**It is required to specify** the ImageNet data folder (`--data_path`), your experiment name & log dir (`--exp_name` and `--exp_dir`, automatically created if not exists), the model name (`--model`, valid choices see the keys of 'HP_DEFAULT_VALUES' in [/downstream_imagenet/arg.py line14](/downstream_imagenet/arg.py#L14)), and the pretrained weight file `--resume_from` to run fine-tuning.

All the other configurations have their default values, listed in [/downstream_imagenet/arg.py#L13](/downstream_imagenet/arg.py#L13).
You can overwrite any defaults by `--bs=1024` or something like that.


Here is an example to pretrain a ConvNeXt-Small on an 8-GPU single machine:
```shell script
$ cd /path/to/SparK/downstream_imagenet
$ torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=<some_port> main.py \
  --data_path=/path/to/imagenet --exp_name=<your_exp_name> --exp_dir=/path/to/logdir \
  --model=convnext_small --resume_from=/some/path/to/convnextS_1kpretrained_official_style.pth
```

For multiple machines, change the `--nnodes` and `--master_addr` to your configurations. E.g.:
```shell script
$ torchrun --nproc_per_node=8 --nnodes=<your_nnodes> --node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port> main.py \
  ...
```


## Logging

See files under `--exp_dir` to track your experiment:

- `<model>_1kfinetuned_last.pth`: the latest model weights
- `<model>_1kfinetuned_best.pth`: model weights with the highest acc
- `<model>_1kfinetuned_best_ema.pth`: EMA weights with the highest acc
- `finetune_log.txt`: records some important information such as:
    - `git_commit_id`: git version
    - `cmd`: all arguments passed to the script
    
    It also reports training loss/acc, best evaluation acc, and remaining time at each epoch.

- `tensorboard_log/`: saves a lot of tensorboard logs, you can visualize accuracies, loss values, learning rates, gradient norms and more things via `tensorboard --logdir /path/to/this/tensorboard_log/ --port 23333`.

## Resuming

Use `--resume_from` again, like `--resume_from=path/to/<model>_1kfinetuned_last.pth`.
