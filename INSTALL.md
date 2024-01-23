# Preparation for pre-training & ImageNet fine-tuning

## Pip dependencies

1. Prepare a python environment, e.g.:
```shell script
$ conda create -n spark python=3.8 -y
$ conda activate spark
```

2. Install `PyTorch` and `timm` (better to use `torch~=1.10`, `torchvision~=0.11`, and `timm==0.5.4`) then other python packages:
```shell script
$ pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install timm==0.5.4
$ pip install -r requirements.txt
```

It is highly recommended to install these versions to ensure a consistent environment for re-implementation.


## ImageNet preparation

Prepare the [ImageNet-1k](http://image-net.org/) dataset
- assume the dataset is in `/path/to/imagenet`
- it should look like this:
```
/path/to/imagenet/:
    train/:
        class1: 
            a_lot_images.jpeg
        class2:
            a_lot_images.jpeg
    val/:
        class1:
            a_lot_images.jpeg
        class2:
            a_lot_images.jpeg
```
- that argument of `--data_path=/path/to/imagenet` should be passed to the training script introduced later 


> `PS:` In our implementation, we use pytorch built-in operators to simulate the submanifold sparse convolution in [encoder.py](https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py) for generality,
due to the fact that many convolution operators (e.g., grouped conv and dilated conv) do not yet have efficient sparse implementations on today's hardware.
If you want to try those sparse convolution, you may refer to [this](https://github.com/facebookresearch/SparseConvNet) sparse convolution library or [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
