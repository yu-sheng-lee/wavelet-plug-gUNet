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
└─ data
    ├─ ITS
    │   ├─ GT
    │   │   └─ ... (image filename)
    │   └─ IN
    │       └─ ... (corresponds to the former)
    └─ ... (dataset name)
```

## Training and Evaluation

### Train

You can modify the training settings for each experiment in the `configs` folder.
Then run the following script to train the model:

```sh
torchrun --nproc_per_node=4 train.py --model (model name) --train_set (train subset name) --val_set (valid subset name) --exp (exp name) --use_mp --use_ddp
```

For example, we train the gUNet-B on the ITS:

```sh
torchrun --nproc_per_node=4 train.py --model gunet_b --train_set ITS --val_set SOTS-IN --exp reside-in --use_mp --use_ddp
```

Note that we use mixed precision training and distributed data parallel by default.

### Test

Run the following script to test the trained model:

```sh
python test.py --model (model name) --test_set (test subset name) --exp (exp name)
```

For example, we test the gUNet-B on the SOTS indoor set:

```sh
python test.py --model gunet_b --test_set SOTS-IN --exp reside-in
```

All test scripts can be found in `run.sh`.

### Overhead

Run the following script to compute the overhead:

```sh
python overhead.py --model (model name)
```

For example, we compute the #Param / MACs / Latency of gUNet-B:

```sh
python overhead.py --model gunet-b
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{song2022vision,
  title={Rethinking Performance Gains in Image Dehazing Networks},
  author={Song, Yuda and Zhou, Yang and Qian, Hui and Du, Xin},
  journal={arXiv preprint arXiv:2209.11448},
  year={2022}
}
```
