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
        self.probabilistic = True
        super(Model, self).__init__(args)
        self.r_mu = nn.Parameter(torch.zeros(args.num_classes,args.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(args.num_classes,args.z_dim))
        self.C = nn.Parameter(torch.ones([]))
        self.optim.add_param_group({'params':[self.r_mu,self.r_sigma,self.C],'lr':self.lr,'momentum':0.9})

    def train_client(self,loader,steps=1):
        self.train()
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        regL2RMeter = AverageMeter()
        regCMIMeter = AverageMeter()
        regNegEntMeter = AverageMeter()
        for step in range(steps):
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            z, (z_mu,z_sigma) = self.featurize(x,return_dist=True)
            logits = self.cls(z)
            loss = F.cross_entropy(logits,y)

            obj = loss
            regL2R = torch.zeros_like(obj)
            regCMI = torch.zeros_like(obj)
            regNegEnt = torch.zeros_like(obj)
            if self.L2R_coeff != 0.0:
                regL2R = z.norm(dim=1).mean()
                obj = obj + self.L2R_coeff*regL2R
            if self.CMI_coeff != 0.0:
                r_sigma_softplus = F.softplus(self.r_sigma)
                r_mu = self.r_mu[y]
                r_sigma = r_sigma_softplus[y]
                z_mu_scaled = z_mu*self.C
                z_sigma_scaled = z_sigma*self.C
                regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                        (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
                regCMI = regCMI.sum(1).mean()
                obj = obj + self.CMI_coeff*regCMI

            z_dist = distributions.Independent(distributions.normal.Normal(z_mu,z_sigma),1)
            mix_coeff = distributions.categorical.Categorical(x.new_ones(x.shape[0]))
            mixture = distributions.mixture_same_family.MixtureSameFamily(mix_coeff,z_dist)
            log_prob = mixture.log_prob(z)
            regNegEnt = log_prob.mean()


            self.optim.zero_grad()
            obj.backward()
            self.optim.step()

            acc = (logits.argmax(1)==y).float().mean()
            lossMeter.update(loss.data,x.shape[0])
            accMeter.update(acc.data,x.shape[0])
            regL2RMeter.update(regL2R.data,x.shape[0])
            regCMIMeter.update(regCMI.data,x.shape[0])
            regNegEntMeter.update(regNegEnt.data,x.shape[0])

        return {'acc': accMeter.average(), 'loss': lossMeter.average(), 'regL2R': regL2RMeter.average(), 'regCMI': regCMIMeter.average(), 'regNegEnt': regNegEntMeter.average()}

