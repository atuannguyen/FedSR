# FedSR: A Simple and Effective Domain Generalization Method for Federated Learning

This repository is the official implementation for the NeurIPS 2022 paper [FedSR: A Simple and Effective Domain Generalization Method for Federated Learning](https://openreview.net/pdf?id=mrt90D00aQX).

Please consider citing our paper as

```
@inproceedings{
nguyen2022fedsr,
title={FedSR: A Simple and Effective Domain Generalization Method for Federated Learning},
author={A. Tuan Nguyen and Philip Torr and Ser-Nam Lim},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=mrt90D00aQX}
}
```

## Credits:

Code for DG datasets is modified from https://github.com/facebookresearch/DomainBed

## Requirements:
python3, pytorch 1.7.0 or higher, torchvision 0.8.0 or higher

## How to run:

Currently, the implementation uses a distributed system with N gpus (with N equals the number of domains). This is to mimic a real-world system. Therefore, the code can't run if you has < N gpus. I will consider adding support for this in the future.

For example, to run the experiment for PACS with target domain 0:

```
cd src

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port $RANDOM -m \
    main --dataset PACS --test_env 0 --method [method] --total_iters 5000 --optim SGD \
    --back_bone resnet18 --train_split 0.9 --z_dim 512 --L2R_coeff 0.01 --CMI_coeff 0.001 \
    --num_samples 20 --seed [seed] --rounds_per_eval 10 --E 5 --batchsize 64 --lr 0.01 \
    --weight_decay 0.0005 --dataset_folder [data_dir] --experiment_path [experiment_path] \
    --save_checkpoint True --distributed True --world_size 4
```

Where:
- [method] is either FedSR or FedL2R (FedL2R is the variant with a deterministic representation and only uses the L2R regularizer)
- [seed] is the random seed (0,1,2).
- [data_dir] is the /path/to/your/data/directory
- [experiment_path] is /path/to/experiment/folder where you save the checkpoints and such

For OfficeHome and DomainNet: change [--back_bone] to resnet50, [--z_dim] to 2048, set [--nproc_per_node] and [--world_size] to the corresponding number of domains (4 and 6). Also, change [--L2R_coeff] and [--CMI_coeff] to the hyper-parameters stated in the paper.

