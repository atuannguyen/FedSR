import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributions as dist

import copy
import numpy as np
from collections import defaultdict, OrderedDict

from methods.base import *
from util import *


class Model(Base):
    def __init__(self, args):
        self.probabilistic = False
        super(Model, self).__init__(args)

    def train_client(self,loader,steps=1):
        self.train()
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        regMeter = AverageMeter()
        for step in range(steps):
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            z = self.featurize(x)
            logits = self.cls(z)
            loss = F.cross_entropy(logits,y)
            reg = z.norm(dim=1).mean()
            self.optim.zero_grad()
            (loss+reg*self.L2R_coeff).backward()
            self.optim.step()

            acc = (logits.argmax(1)==y).float().mean()
            lossMeter.update(loss.data,x.shape[0])
            accMeter.update(acc.data,x.shape[0])
            regMeter.update(reg.data,x.shape[0])

        return {'acc': accMeter.average(), 'loss': lossMeter.average(), 'reg': regMeter.average()}

