## Installation

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


3. Prepare the [ImageNet-1k](http://image-net.org/) dataset
    - assume the dataset is in `/path/to/imagenet`
    - check the file path, it should look like this:
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


4. (Optional) Install [this](https://github.com/facebookresearch/SparseConvNet) sparse convolution library:
```shell script
$ git clone https://github.com/facebookresearch/SparseConvNet.git && cd SparseConvNet
$ rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
$ python3 setup.py develop --user
```


> `Tips:` In our default implementation, we use pytorch builtin operators to simulate the submanifold sparse convolution in [encoder.py](https://github.com/keyu-tian/SparK/blob/main/encoder.py) for generality,
due to the fact that many convolution operators (e.g., grouped conv and dilated conv) do not yet have efficient sparse implementations on today's hardware.
If you would like to use the *true* sparse convolution installed above, please pass `--sparse_conv=1` to the training script, but it would be much slower.
