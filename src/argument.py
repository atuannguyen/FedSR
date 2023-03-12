import argparse
import numpy as np
import hashlib

parser = argparse.ArgumentParser(description='Exp',conflict_handler='resolve')

parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--test_env', type=int, default=0)
parser.add_argument('--method', type=str, default='FedSR')
parser.add_argument('--total_iters', type=int, default=5000)
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--back_bone', type=str, default='resnet18', choices=['resnet18','resnet50'])
parser.add_argument('--train_split', type=float, default=0.9)
parser.add_argument('--z_dim', type=int, default=512)
parser.add_argument('--num_samples', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--rounds_per_eval', type=int, default=1)
parser.add_argument('--save_checkpoint', type=str, default='True',choices=['True','False'])
parser.add_argument('--load_unfinished', type=str, default='False',choices=['True','False'])
parser.add_argument('--saved_folder', type=str, default='None')

    
parser.add_argument('--E', type=int, default=5, help='number of local update each round')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--L2R_coeff', type=float, default=1e-2)
parser.add_argument('--CMI_coeff', type=float, default=5e-4)



# must have for distributed code
parser.add_argument('--dataset_folder', type=str, default='/scratch/local/ssd/tuan/data/')
parser.add_argument('--experiment_path', type=str, default='./experiment_folder/')
parser.add_argument('--distributed', type=str, default='True',choices=['True','False'])
parser.add_argument('--world_size', type=int, default=1)
# for running by torch.distributed
parser.add_argument('--rank', type=int, default=0)
# for slurm
parser.add_argument('--local_rank', type=int, default=0)

# exclude unimportant args when saving the args
unimportant_args = ['save_checkpoint','load_unfinished','saved_folder','dataset_folder','experiment_path','distributed','world_size','rank','local_rank','unimportant_args']
