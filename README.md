## Getting started

### Install

We test the code on PyTorch 2.2.1 + CUDA 11.8 

1. Create a new conda environment
```
conda create -n gunet python=3.11
conda activate gunet
```

2. Install dependencies
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Download


The final file path should be the same as the following:

```
┬─ save_models
│   ├─ reside-in
│   │   ├─ gunet-t.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data_dir
    ├─ dataset 1
    │   ├─ GT
    │   │   └─ ... (image filename)
    │   └─ hazy
    │       └─ ... (corresponds to the former)
    ├─ dataset 2
    │   ├─ GT
    │   │   └─ ... (image filename)
    │   └─ hazy
    │       └─ ... (corresponds to the former)
    
    └─ ... (dataset name)
```
The weights of Reside-out dataset can be downloaded from the following website.Please unzip the file and place it in the saved_models folder.
https://drive.google.com/file/d/14IqHDOWIkTExzzlCXLK7NjeU3y_g59IW/view?usp=drive_link

The scores for the Reside-out dataset are shown in the table below.

wavelet:

| Dehazing model | Params(M) | FLOPS(G) | PSNR | SSIM |
| ---------- | --------- | --------- | --------- | ---------|
| two oder (wavelet_gnet_two) | 538,256 | 925.29(M) | 32.14 | 0.98 |
| Three oder (wavelet_gnet_three) | 612,128 | 576.86(M) | 29.08 | 0.97 |

plug(ll_predict)

| Dehazing model | Params(M) | FLOPS(G) | PSNR | SSIM |
| -------- | ------- | ------- | ------- | --------|
| gunet_ss | 524,197 | 1.01 | 33.20 | 0.98 |
| gunet_t | 853,237 | 1.37 |34.38 | 0.9 |

lite plug(ll_predict_lite)

| Dehazing model | Params(M) | FLOPS(G)  | PSNR  | SSIM |
| -------- | ------- |-----------|-------|------|
| gunet_ss | 520,639 | 613.42(M) | 31.92 | 0.97 |
| gunet_t | 849,679 | 979.48(M) | 33.14 | 0.98 |


## Training and Evaluation

### Train

You can modify the training settings for each experiment in the `configs` folder.
Then run the following script to train the model:

```sh
python train.py --model (model name) --ll_predict_model (When using plug, need to behazing model backbone) --data_dir (dataset root path) --train_set (train subset name) --val_set (valid subset name) --exp (exp name) 
```
The following are the models that can be used.
"gunet_ss",'gunet_t', 'gunet_s', 'gunet_b', 'gunet_d','wavelet_gnet'','wavelet_gnet_two','wavelet_gnet_three'
,For example, we train the gUNet-B on the reside-in dataset:

```sh
python train.py --model gunet_b --data_dir <path_to_dataset> --train_set ITS --val_set SOTS-IN --exp reside-in
```
If you want to use the plugin module, you can use two models, 'll_predict' and 'll_predict_lite', and specify the backbone.
,For example, we train the plugin module with gUNet-T on the reside-in dataset:
```sh
python train.py --model ll_predict --ll_predict_model gunet_t --data_dir <path_to_dataset> --train_set ITS --val_set SOTS-IN --exp reside-in
```

Note that we use mixed precision training by default.

### Test

Run the following script to test the trained model，It is almost identical to training.:

```sh
python test.py --model (model name) --ll_predict_model (When using plug, need to behazing model backbone) --data_dir <path_to_dataset> --test_set (test subset name) --exp (exp name) 
```

For example, we test the gUNet-B on the SOTS indoor set:

```sh
python test.py --model gunet_b --data_dir <path_to_dataset> --test_set SOTS-IN --exp reside-in
```