import os
import torch
import random
import numpy as np
import json
import hashlib
import socket
import time
from util import *
from argument import *



args = parser.parse_args()

for name in args.__dict__:
    if getattr(args,name) in ['True','False','None']:
        setattr(args,name,eval(getattr(args,name)))
    if callable(getattr(args,name)):
        setattr(args,name,getattr(args,name)(args.hparams_gen_seed))

important_args = {k: getattr(args,k) for k in args.__dict__ if k not in unimportant_args}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

args.gpus_per_node = torch.cuda.device_count()


if args.distributed:
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        args.rank = args.rank + args.local_rank #base rank + local_rank
        args.gpu = args.local_rank % torch.cuda.device_count()

    # Set some environment flags
    if "SLURM_JOBID" in os.environ:
        jobid = os.environ["SLURM_JOBID"]
    else:
        jobid = '-1'
    hostfile = os.path.join(args.experiment_path, jobid  + ".txt")

    if args.rank == 0:
        ip = socket.gethostbyname(socket.gethostname())
        if 'MASTER_PORT' in os.environ:
            port = os.environ['MASTER_PORT']
        else:
            port = find_free_port()
        endpoint = '{}{}:{}'.format('tcp://', ip, port)
        os.environ['MASTER_ADDR'] = endpoint
        with open(hostfile, "w") as f:
            f.write(endpoint)
    else:
        while not os.path.exists(hostfile):
            time.sleep(1)
        with open(hostfile, "r") as f:
            os.environ['MASTER_ADDR'] = f.read().strip()
    print(args.rank, args.world_size,os.environ['MASTER_ADDR'])

    torch.distributed.init_process_group(backend='nccl', init_method=os.environ['MASTER_ADDR'],
                           world_size=args.world_size, rank=args.rank)
    try:
        os.remove(hostfile)
    except:
        pass

    torch.cuda.set_device(args.gpu)


else:
    args.gpu = None


print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
