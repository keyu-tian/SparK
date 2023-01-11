## Pre-training from scratch

The script for pre-training is [exp/pt.sh](https://github.com/keyu-tian/SparK/blob/main/scripts/pt.sh).
Since `torch.nn.parallel.DistributedDataParallel` is used for distributed training, you are expected to specify some distributed arguments on each node, including:
- `--num_nodes=<INTEGER>`
- `--ngpu_per_node=<INTEGER>`
- `--node_rank=<INTEGER>`
- `--master_address=<ADDRESS>`
- `--master_port=<INTEGER>`

Set `--num_nodes=0` if your task is running on a single GPU.


You can add arbitrary key-word arguments (like `--ep=400 --bs=2048`) to specify some pre-training hyperparameters (see [utils/meta.py](https://github.com/keyu-tian/SparK/blob/main/utils/meta.py) for all).

Here is an example command:
```shell script
$ cd /path/to/SparK
$ bash ./scripts/pt.sh <experiment_name> \
--num_nodes=1 --ngpu_per_node=8 --node_rank=0 \
--master_address=128.0.0.0 --master_port=30000 \
--data_path=/path/to/imagenet \
--model=res50 --ep=1600 --bs=4096
```

Note that the first argument is the name of experiment.
It will be used to create the output directory named `output_<experiment_name>`.


## Logging

Once an experiment starts running, the following files would be automatically created and updated in `SparK/output_<experiment_name>`:

- `ckpt-last.pth`: includes model states, optimizer states, current epoch, current reconstruction loss, etc.
- `log.txt`: records important meta information such as:
    - the git version (commid_id) at the start of the experiment
    - all arguments passed to the script
    
    It also reports the loss and remaining training time at each epoch.

- `stdout_backup.txt` and `stderr_backup.txt`: will save all output to stdout/stderr

We believe these files can help trace the experiment well.


## Resuming

To resume from a saved checkpoint, run `pt.sh` with `--resume=/path/to/checkpoint.pth`.



## Regarding sparse convolution

For generality, we use the masked convolution implemented in [encoder.py](https://github.com/keyu-tian/SparK/blob/main/encoder.py) to simulate submanifold sparse convolution by default.
If `--sparse_conv=1` is not specified, this masked convolution would be used in pre-training.

**For anyone who might want to run SparK on another architectures**:
we recommend you to use the default masked convolution, 
given the limited optimization of sparse convolution in hardware, and in particular the lack of efficient implementation of many modern operators like grouped conv and dilated conv.

