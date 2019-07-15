"""
Adapted from https://github.com/meetshah1995/pytorch-semseg
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    weight_tensor = torch.Tensor([1, weight]).cuda()
    log_p = F.log_softmax(input, dim=1) # F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]

    loss = F.nll_loss(log_p, target, weight=weight_tensor, reduce=False) # reduce=False
    
    if size_average:
        loss /= mask.data.sum()

    return loss.sum()


def cross_entropy3d(input, target, weight=None, size_average=True):
    n, c, h, w, d = input.size()
    weight_tensor = torch.Tensor([1, weight, weight, weight, weight]).cuda()
    log_p = F.log_softmax(input.view(n, -1))
    log_p = log_p.view(input.size())
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w * d, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]

    loss = F.nll_loss(log_p, target, weight=weight_tensor)

    if size_average:
        loss /= mask.data.sum()

    return loss.sum()


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)
