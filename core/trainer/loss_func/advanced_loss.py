# Author: lqxu

import torch
from torch import nn, Tensor, LongTensor

__all__ = ["focal_loss", "multi_label_cross_entropy_loss"]


def focal_loss(logits, target, weight=None, gamma=2., ignore_index=-100, reduction="mean"):
    # type: (Tensor, LongTensor, Tensor, float, int, str) -> Tensor
    """ 多分类版本的 focal loss """
    log_probs = nn.functional.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    loss = (1 - probs) ** gamma * log_probs
    loss = nn.functional.nll_loss(loss, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    return loss


def multi_label_cross_entropy_loss(logits, target, reduction="mean"):
    # type: (Tensor, Tensor, str) -> Tensor
    """ 苏剑林版本的多标签分类任务 loss \n

    注意: 这里的 mask 指的是 sequence mask, 类型是 BoolTensor, 不是 FloatTensor \n
    注意: mask 和 BERT 的 mask 是不一致的, 需要 mask 掉的是 True (不参与计算的), 不需要 mask 掉的是 False (参与计算的) \n

    reference: https://kexue.fm/archives/7359 \n
    """
    # ## step 1: 计算目标类的 logits 值 (将目标类的 logits 值变成负数, 非目标类的 logits 值变成 -inf)
    # 如果 target = 1, 那么 target_logits = -logits, 如果 target = 0, 那么 target_logits = -inf
    target_logits = -logits - (1 - target) * 10000.

    # ## step 2: 获得非目标类的 logits 值 (将目标类的 logits 值变成 -inf, 非目标类的 logits 值不变)
    # 如果 target = 1, 那么 non_target_logits = -inf, 如果 target = 0, 那么 non_target_logits = logits
    non_target_logits = logits - target * 10000.

    # ## step 3: 给 target_logits 和 non_target_logits 添加一个 0
    target_logits = nn.functional.pad(input=target_logits, pad=(0, 1))  # [batch_size, num_tags+1]
    non_target_logits = nn.functional.pad(input=non_target_logits, pad=(0, 1))  # [batch_size, num_tags+1]

    # ## step 4: 计算 loss 值
    loss = torch.logsumexp(target_logits, dim=-1) + torch.logsumexp(non_target_logits, dim=-1)  # [batch_size, ]

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unsupported reduction method of {reduction}")
