## About code isolation

This `downstream_imagenet` is isolated from pre-training codes. One can treat this `downstream_imagenet` as an independent codebase 🛠️.


## Preparation for ImageNet-1k fine-tuning

See [INSTALL.md](https://github.com/keyu-tian/SparK/blob/main/INSTALL.md) to prepare `pip` dependencies and the ImageNet dataset.

**Note: for network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**


## Fine-tuning on ImageNet-1k from pre-trained weights

Run [downstream_imagenet/main.sh](https://github.com/keyu-tian/SparK/blob/main/downstream_imagenet/main.sh).
It is **required** to specify your experiment_name `<experiment_name>` ImageNet data folder `--data_path`, model name `--model`, and checkpoint file path `--resume_from` to run fine-tuning.
All the other configurations have their default values, listed in [downstream_imagenet/arg.py#L13](https://github.com/keyu-tian/SparK/blob/main/downstream_imagenet/arg.py#L13).
You can override any defaults by passing key-word arguments (like `--bs=2048`) to `main.sh`.


Here is an example command fine-tuning a ResNet50 on single machine with 8 GPUs:
```shell script
$ cd /path/to/SparK/downstream_imagenet
$ bash ./main.sh <experiment_name> \
  --num_nodes=1 --ngpu_per_node=8 \
  --data_path=/path/to/imagenet \
  --model=resnet50 --resume_from=/some/path/to/timm_resnet50_1kpretrained.pth
```

For multiple machines, change the `num_nodes` to your count and plus these args:
```shell script
--node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port>
```

Note that the first argument `<experiment_name>` is the name of your experiment, which would be used to create an output directory named `output_<experiment_name>`.


## Logging

Once an experiment starts running, the following files would be automatically created and updated in `output_<experiment_name>`:

- `<model>_1kfinetuned_last.pth`: the latest model weights
- `<model>_1kfinetuned_best.pth`: model weights with the highest acc
- `<model>_1kfinetuned_best_ema.pth`: EMA weights with the highest acc
- `finetune_log.txt`: records some important information such as:
    - `git_commit_id`: git version
    - `cmd`: all arguments passed to the script
    
    It also reports training loss/acc, best evaluation acc, and remaining time at each epoch.

These files can help trace the experiment well.


## Resuming

Add `--resume_from=path/to/<model>_1kfinetuned_last.pth` to resume from a latest saved checkpoint.
