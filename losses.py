# -*- coding: utf-8 -*-

import torch
from torch import nn


def rank_reg(pred, eps=0.05):
    try:
        pred_s = torch.linalg.svdvals(pred)  # pred_u, pred_s, pred_v = torch.svd(pred)##
        return torch.sum(pred_s.gt(eps)) / pred_s.numel()
    except Exception as e:
        return 0.0


def percept_loss(pred, gt, fe_layer, criterion=nn.L1Loss(), eps=0.05,
                 weight_rank=0.0, weight_l1=1.0, weight_percept=0.0):
    assert pred.size() == gt.size()
    assert pred.size()[1] == 1 or pred.size()[1] == 3
    
    fe_layer.eval()
    if pred.size()[1] == 1:
        fea_pred = fe_layer(torch.cat([(pred+1.0)/2.0, (pred+1.0)/2.0, (pred+1.0)/2.0], 1))
        fea_gt = fe_layer(torch.cat([(gt+1.0)/2.0, (gt+1.0)/2.0, (gt+1.0)/2.0], 1))
    else:
        fea_pred = fe_layer((pred+1.0)/2.0)
        fea_gt = fe_layer((gt+1.0)/2.0)
    
    loss = weight_l1 * criterion(pred, gt) + weight_rank * rank_reg(pred, eps=eps)
    for i in range(len(fea_gt)):
        loss += weight_percept * criterion(fea_pred[i], fea_gt[i])
    
    return loss
