import os 
import random
import sys 
import importlib
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from datetime import datetime
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np

from datasets import *
from util import *
from prepare import *

data = eval(args.dataset)(root=args.dataset_folder,test_envs=[args.test_env])
args.num_classes = data.num_classes



algo = importlib.import_module('methods.'+args.method).Model(args=args).to(device)

start_round = 0
hist = []
if args.load_unfinished and args.saved_folder is None:
    if os.path.exists(os.path.join(args.experiment_path,'checkpoint.pt')):
        args.saved_folder = args.experiment_path
if args.saved_folder is not None:
    saved_dict = torch.load(os.path.join(args.saved_folder,'checkpoint.pt'))
    start_round = saved_dict['round'] + 1
    hist = torch.load(os.path.join(args.saved_folder,'hist'))['hist']
    algo.load_state_dict(saved_dict['state_dict'])



if args.rank != args.test_env:
    dataset = data.datasets[args.rank]
    train_len = int(len(dataset) * args.train_split)
    test_len = len(dataset) - train_len
    trainset, testset = random_split(dataset, [train_len,test_len], 
                                generator=torch.Generator().manual_seed(args.seed))
    # change the transform of test split 
    if hasattr(testset.dataset,'transform'):
        import copy
        testset.dataset = copy.copy(testset.dataset)
        testset.dataset.transform = data.transform

    trainloader = DataLoader(trainset,batch_size=args.batchsize,shuffle=True,num_workers=4)
    testloader = DataLoader(testset,batch_size=args.batchsize,shuffle=True,num_workers=4)
else:
    testset = data.datasets[args.rank]
    testloader = DataLoader(testset,batch_size=args.batchsize,shuffle=True,num_workers=4)



rounds = int(args.total_iters/args.E)
algo.communicate(master=args.test_env)
best_acc = 0.0
best_acc_by_source = 0.0
best_acc_source = 0.0
best_loss_source = float('inf')
for round in range(start_round,rounds):
    # Train on the source domains (clients)
    if args.rank != args.test_env:
        r = algo.train_client(trainloader,steps=args.E)
        s = ''
        for key in r:
            s += '{} {} '.format(key,r[key].float())
    
    # Update in the server and Test target domain
    algo.communicate(master=args.test_env)
    
    if (round%args.rounds_per_eval==0 or round==rounds-1):
        print(f"Round {round}/{rounds}: {datetime.now()}")
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        algo.eval()
        for x,y in testloader:
            x,y = x.to(device), y.to(device)
            preds = algo(x)
            acc = (torch.argmax(preds,1)==y).float().mean()
            loss = F.cross_entropy(preds,y)
            acc_meter.update(acc.detach(),x.shape[0])
            loss_meter.update(loss.detach(),x.shape[0])

        accs = all_gather(acc_meter.average(),stack=False)
        losses = all_gather(loss_meter.average(),stack=False)
        accs = [x.item() for x in accs]
        losses = [x.item() for x in losses]
        if args.rank == args.test_env:
            source_accs = accs[:args.test_env] + accs[args.test_env+1:]
            target_acc = accs[args.test_env]
            source_losses = losses[:args.test_env] + losses[args.test_env+1:]
            target_loss = losses[args.test_env]
            r = {'round': round,
                'source_accs': source_accs,
                'target_acc': target_acc,
                'source_losses': source_losses,
                'target_loss': target_loss}
            hist.append(r)
            torch.save({'hist': hist, 'args': important_args}, os.path.join(args.experiment_path,'hist'))

            if args.save_checkpoint:
                save_dict = {'state_dict':algo.state_dict(),
                            'round':round}
                torch.save(save_dict, os.path.join(args.experiment_path,'checkpoint.pt'))

            best_acc = max(best_acc,accs[args.test_env])
            source_acc = sum(accs) - accs[args.test_env]
            source_loss = sum(losses) - losses[args.test_env]
            
            if source_loss <= best_loss_source:
                best_acc_by_source = accs[args.test_env]
                best_loss_source = source_loss
            print(f'Target domain {args.rank}: Test acc {acc_meter.average()}, Best by source {best_acc_by_source}, Best by target {best_acc}\n')
        else:
            print(f'Source domain {args.rank}: Training {s},   Test acc {acc_meter.average()}')

mark_done(args.experiment_path)
