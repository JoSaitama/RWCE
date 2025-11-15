import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from conformal_learning.utils import calc_true_label_rank
from conformal_learning.help import *


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

# class IWCELoss(nn.Module):
#     def __init__(self, weight=None, reduction='mean', ags = None):
#         super(IWCELoss, self).__init__()
#         self.weight = weight
#         self.reduction = reduction
#         self.ags = ags
#         self.print_interval = 100

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor):
#         if self.training and (logits.shape[0] % self.print_interval == 0):
#            print(f"[IWCE] Forward pass executing... logits shape: {logits.shape}, targets shape: {targets.shape}")
#         #ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
#         ce_loss = nn.CrossEntropyLoss(weight=self.weight)
#         print(f"Cross Entropy Loss (individual): {ce_loss}")
#         rank = calc_true_label_rank(logits, targets).float()
#         print(f"[IWCE] True label rank (default): {rank}")

#         iwce = ce_loss * rank
#         print(f"[IWCE] IWCE loss: {iwce}")     
#         if self.reduction == 'mean':
#             return iwce.mean()
#         elif self.reduction == 'sum':
#             return iwce.sum()
#         else:
#             return iwce

class IWCELoss(nn.Module):
    
    def __init__(self, weight=None, reduction='mean', ags=None, device=None):
        super(IWCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ags = ags
        self.device = device
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.weight, reduction='none')

    def compute_scores(self, proba, targets):
        # from conformal_learning.black_boxes_CNN import (
        #     find_scores_APS, find_scores_RAPS, find_scores_HPS
        # ) 
        method = self.ags.train_CP_score if hasattr(self.ags, 'train_CP_score') else 'HPS'

        # if method == 'APS':
        #     scores = find_scores_APS(proba, targets, device=self.device)
        # elif method == 'RAPS':
        #     scores = find_scores_RAPS(proba, targets, device=self.device)
        # else:  # default to HPS
        #     scores = find_scores_HPS(proba, targets, device=self.device)
        # return scores
        if method == 'APS':
            scores = compute_scores_diff(proba, targets, device=self.device)
        elif method == 'RAPS':
            scores = compute_scores_diff_RAPS(proba, targets, device=self.device)
        else:  # default to HPS
            scores = compute_HPS_scores(proba, targets, device=self.device)
        return scores

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # print(f"[DEBUG] reduction: {self.reduction}")
        # Compute conformity scores and rank

        Temp = max(1e-5, self.ags.train_T)
        logits = logits / Temp
        proba = torch.softmax(logits, dim=1)
        
        scores = self.compute_scores(proba, targets)
        # print(f"[DEBUG] scores (first 5): {scores[:5]}")

        ranks = calc_true_label_rank(scores, targets).float()
        # print(f"[DEBUG] ranks (first 5): {ranks[:5]}")

        # Cross entropy loss (no reduction)
        ce_losses = self.ce_loss_fn(logits, targets)
        # print(f"[DEBUG] CE losses (first 5): {ce_losses[:5]}")

        # Importance weighted CE
        iwce = ce_losses * ranks
        # print(f"[DEBUG] IWCE losses (first 5): {iwce[:5]}")

        if self.reduction == 'mean':
            return iwce.mean()
        elif self.reduction == 'sum':
            return iwce.sum()
        else:
            return iwce  # no reduction