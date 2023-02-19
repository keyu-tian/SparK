## Preparation for ImageNet-1k fine-tuning

See [INSTALL.md](https://github.com/keyu-tian/SparK/blob/main/INSTALL.md) to prepare dependencies and ImageNet dataset.

**Note: for network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**


## Pre-training on ImageNet-1k from scratch

Run [main.sh](https://github.com/keyu-tian/SparK/blob/main/main.sh).

It is **required** to specify ImageNet data folder and model name to run pre-training.
Besides, you can pass arbitrary key-word arguments (like `--ep=400 --bs=2048`) to `main.sh` to specify some pre-training hyperparameters (see [utils/arg_utils.py](https://github.com/keyu-tian/SparK/blob/main/utils/arg_utils.py) for all hyperparameters and their default values).


Here is an example command pre-training a ResNet50 on single machine with 8 GPUs:
```shell script
$ cd /path/to/SparK
$ bash ./main.sh <experiment_name> \
  --num_nodes=1 --ngpu_per_node=8 \
  --data_path=/path/to/imagenet \
  --model=resnet50 --ep=1600 --bs=4096
```

For multiple machines, change the `num_nodes` to your count and plus these args:
```shell script
--node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port>
```

Note that the first argument `<experiment_name>` is the name of your experiment, which would be used to create an output directory named `output_<experiment_name>`.


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

For generality, we use the masked convolution implemented in [encoder.py](https://github.com/keyu-tian/SparK/blob/main/encoder.py) to simulate submanifold sparse convolution by default.
<!--If `--sparse_conv=1` is not specified, this masked convolution would be used in pre-training.-->

**For anyone who might want to run SparK on another architectures**:
we recommend you to use the default masked convolution, 
considering the limited optimization of sparse convolution on hardwares, and in particular the lack of efficient implementation of many modern operators like grouped conv and dilated conv.
