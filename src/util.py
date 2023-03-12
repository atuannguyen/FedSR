import subprocess
import torch.distributed as dist
import torch
import numpy as np
import os

def all_gather(tensor,stack=True):
    if not isinstance(tensor,torch.Tensor):
        tensor = torch.tensor(tensor).cuda()
    if torch.distributed.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list,tensor.contiguous())
        tensor_list[rank] = tensor
        if not stack:
            return tensor_list
        if len(tensor.shape) == 0:
            tensor = torch.stack(tensor_list).mean()
        else:
            tensor = torch.cat(tensor_list)
    return tensor

def parallelize(model, distributed=True, gpu=0):
    if distributed:
        model  = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        return torch.nn.DataParallel(model)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.sum = 0
    def update(self,val,n=1):
        self.count += n
        self.sum += val*n
    def average(self):
        return self.sum/self.count
    def __repr__(self):
        r = self.sum/self.count
        if r<1e-3:
            return '{:.2e}'.format(r)
        else:
            return '%.4f'%(r)


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def check_nvidia():
    cmd = 'nvidia-smi'
    stdout = subprocess.check_output(cmd.split())
    stdout = stdout.decode('UTF-8') 
    return stdout

def mark_done(path):
    f = open(os.path.join(path,'done'), "w")
    f.write("Done")
    f.close()
