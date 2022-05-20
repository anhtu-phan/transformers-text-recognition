# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py

import torch
import torch.nn.functional as F


def cal_performance(pred, gold, trg_pad_idx, smoothing=False, eps=0.1):
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing, eps)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False, eps=0.1):
    gold = gold.contiguous().view(-1)
    if smoothing:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = - (one_hot * log_prob).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')

    return loss
