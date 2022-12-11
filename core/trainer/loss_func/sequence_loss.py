# Author: lqxu

"""
这个模块用于 token 级别的 loss, 为各种 loss 添加 mask 操作

里面没有 CrossEntropyLoss, 因为需要将 padding 部分的 target 变成 -100 即可
"""

import torch
from torch import nn, Tensor

__all__ = ["multi_label_cross_entropy_loss_with_mask", "binary_cross_entropy_with_logits_and_mask"]


def multi_label_cross_entropy_loss_with_mask(logits, target, cal_mask):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    苏剑林版本的多标签分类任务 loss, 将目标类和非目标类分开计算, 进行了以下改动:
        1. 使用 torch.where 实现, 代码更加容易读
        2. 添加序列 mask 支持
        3. 删除对于 reduction 参数的支持

    logits 和 target 的 shape 应该是相同的, eg. [batch_size, num_tokens, num_labels] \n
    cal_mask 的 shape 应该比 logits 少一个维度, eg. [batch_size, num_tokens] \n

    reference: https://kexue.fm/archives/7359
    """

    target = target.bool()
    cal_mask = cal_mask.bool().unsqueeze(-1)

    # step1: 计算目标类的 logits 值
    # 如果是目标类, 那么 target_logits 值为 -logits, 如果是非目标类, 那么 target_logits 值为 -10000.
    target_logits = torch.where(target & cal_mask, -logits, -10000.)  # [n_samples, n_labels]
    target_logits = nn.functional.pad(input=target_logits, pad=(0, 1))  # [n_samples, n_labels+1]

    # step2: 计算非目标类的 logits 值
    # 如果是非目标类, 那么 non_target_logits 值为 logits, 如果是非目标类, 那么 non_target_logits 值为 -10000.
    non_target_logits = torch.where((~target) & cal_mask, logits, -10000.)  # [n_samples, n_labels]
    non_target_logits = nn.functional.pad(input=non_target_logits, pad=(0, 1))  # [n_samples, n_labels+1]

    # ## step 4: 计算 loss 值
    loss = torch.logsumexp(target_logits, dim=-1) + torch.logsumexp(non_target_logits, dim=-1)  # [n_samples, ]

    return torch.sum(loss) / torch.sum(cal_mask)


def binary_cross_entropy_with_logits_and_mask(logits, target, cal_mask):

    """
    带 mask 的 BCEWithLogitsLoss 函数 \n

    logits 和 target 的 shape 是一致的, 比方说 [batch_size, num_tokens, num_labels] \n
    cal_mask 的 shape 比 logits 少一个维度, 比方说 [batch_size, num_tokens, ]
    """
    cal_mask = cal_mask.unsqueeze(-1).float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        input=logits, target=target, reduction="none"
    )
    loss = torch.sum(loss * cal_mask) / torch.sum(cal_mask)
    return loss
