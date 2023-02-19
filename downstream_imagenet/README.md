## Preparation for ImageNet-1k fine-tuning

See [INSTALL.md](https://github.com/keyu-tian/SparK/blob/main/INSTALL.md) to prepare dependencies and ImageNet dataset.

## Training from pre-trained checkpoint

The script file for ImageNet-1k fine-tuning is [downstream_imagenet/main.sh](https://github.com/keyu-tian/SparK/blob/main/downstream_imagenet/main.sh).
Since `torch.nn.parallel.DistributedDataParallel` is used for distributed training, you are expected to specify some distributed arguments on each node, including:
- `--num_nodes=<INTEGER>`
- `--ngpu_per_node=<INTEGER>`
- `--node_rank=<INTEGER>`
- `--master_address=<ADDRESS>`
- `--master_port=<INTEGER>`

It is required to specify ImageNet data folder, model name, and checkpoint file path to run fine-tuning.
All the other arguments have their default values, listed in [downstream_imagenet/arg.py#L13](https://github.com/keyu-tian/SparK/blob/main/downstream_imagenet/arg.py#L13).
You can override any defaults by adding key-word arguments (like `--bs=2048`) to `main.sh`.

Here is an example command:
```shell script
$ cd /path/to/SparK/downstream_imagenet
$ bash ./main.sh <experiment_name> \
--num_nodes=1 --ngpu_per_node=8 --node_rank=0 \
--master_address=128.0.0.0 --master_port=30000 \
--data_path=/path/to/imagenet \
--model=resnet50 --resume_from=/path/to/resnet50_1kpretrained.pth
```

Note that the first argument `<experiment_name>` is the name of your experiment, which would be used to create an output directory named `output_<experiment_name>`.

