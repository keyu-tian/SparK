## Preparation for pre-training


1. prepare a python environment, e.g.:
```shell script
$ conda create -n spark python=3.8 -y
$ conda activate spark
```


2. install `PyTorch` and `timm` environment (better to use `torch~=1.10`, `torchvision~=0.11`, and `timm==0.5.4`) then other python packages, e.g.:
```shell script
$ pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install timm==0.5.4
$ pip install -r requirements.txt
```

It is highly recommended to follow these instructions to ensure a consistent environment for re-implementation.


3. prepare [ImageNet-1k](http://image-net.org/) dataset
    - download the dataset to a folder `/path/to/imagenet`
    - the file structure should look like:
    ```
    /path/to/imagenet/:
        train/:
            class1: 
                a_lot_images.jpeg
            class2:
                a_lot_images.jpeg
        val/:
            class3: 
                a_lot_images.jpeg
            class4:
                a_lot_images.jpeg
    ```


4. (optional) if want to use sparse convolution rather than masked convolution, please install this [library](https://github.com/facebookresearch/SparseConvNet) and set `--sparse_conv=1` later
```shell script
$ git clone https://github.com/facebookresearch/SparseConvNet.git && cd SparseConvNet
$ rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
$ python3 setup.py develop --user

```


## Pre-training from scratch

1. since `torch.nn.parallel.DistributedDataParallel` is used for distributed training, you are expected to specify some distributed arguments on each node, including:
    - `--num_nodes`
    - `--ngpu_per_node`
    - `--node_rank`
    - `--master_address`
    - `--master_port`


2. besides, you also need to specify the name of experiment and the ImageNet path in the first two arguments, and you may add arbitrary hyperparameter key-words (like `--ep=400 --bs=2048`) for other configurations, and the final command should like this:
```shell script
$ cd /path/to/SparK
$ bash ./scripts/pt.sh \
$ experiment_name /path/to/imagenet \
$ --num_nodes=1 --ngpu_per_node=8 --node_rank=0 \
$ --master_address=128.0.0.0 --master_port=30000 \
$ --model=res50 --ep=400 --bs=2048
```


## Resume

When an experiment starts running, the folder `SparK/<experiment_name>` would be created and record per-epoch checkpoints (e.g., `ckpt-last.pth`) and log files (`log.txt`).

To resume from a checkpoint, specify `--resume=/path/to/checkpoint.pth`.



## Read logs

The `stdout` and `stderr` are also saved in `SparK/<experiment_name>/stdout.txt` and `SparK/<experiment_name>/stderr.txt`.

Note `SparK/<experiment_name>/log.txt` would record the most important information like current loss values and the remaining time.


