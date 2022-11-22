# Author: lqxu

"""
PyTorch 中对于矩阵的 pairwise 计算支持并不好, 这里是仿照 sklearn.metrics.pairwise 模块实现相关的计算 \n
本模块默认所有张量传入的是 **矩阵**, 不会做类型检查, 如果传入的不是 **矩阵**, 用户需要自己保证计算的正确性 \n
以 **paired** 开头的函数是 **一对一** 进行计算的, 以 **pairwise** 开头的函数是 **一一配对** 进行计算的 \n
"""

import torch
from torch import Tensor
# noinspection PyPep8Naming
from torch.nn import functional as F

# from sklearn.metrics import pairwise

__all__ = [
    "paired_cosine_similarity", "pairwise_cosine_similarity", "paired_cosine_distance", "pairwise_cosine_distance",
    "paired_dot_product", "pairwise_dot_product", "paired_distance", "pairwise_distance",
]


def paired_dot_product(input1: Tensor, input2: Tensor, keepdim: bool = False) -> Tensor:
    """
    左矩阵的行向量 和 右矩阵的行向量 一对一点乘 \n
    :param input1: [num_samples, num_features]
    :param input2: [num_samples, num_features]
    :param keepdim: 是否保持相同的维度, 默认 False
    :return: [num_samples]
    """
    return torch.sum(torch.mul(input1, input2), dim=-1, keepdim=keepdim)


def pairwise_dot_product(input1: Tensor, input2: Tensor) -> Tensor:
    """
    左矩阵的行向量 和 右矩阵的行向量 两两配对进行点乘 \n
    :param input1: [num_samples1, num_features]
    :param input2: [num_samples2, num_features]
    :return: [num_samples1, num_samples2]
    """
    return torch.mm(input1, input2.T)


def paired_cosine_similarity(input1: Tensor, input2: Tensor, keepdim: bool = False) -> Tensor:
    ret = torch.nn.functional.cosine_similarity(input1, input2, dim=-1, eps=1e-8)
    if keepdim: ret = torch.unsqueeze(ret, dim=-1)  # noqa: E701
    return ret


def pairwise_cosine_similarity(input1: Tensor, input2: Tensor) -> Tensor:
    # input1 = torch.nn.functional.normalize(input1, p=2, dim=-1, eps=1e-12)
    # input2 = torch.nn.functional.normalize(input2, p=2, dim=-1, eps=1e-12)
    # return pairwise_dot_product(input1, input2, keepdim=False)
    input1 = torch.unsqueeze(input1, dim=1)  # [num_samples1, 1, num_features]
    input2 = torch.unsqueeze(input2, dim=0)  # [1, num_samples2, num_features]
    return F.cosine_similarity(input1, input2, dim=-1, eps=1e-8)  # [num_samples1, num_samples2]


def paired_cosine_distance(input1: Tensor, input2: Tensor, keepdim: bool = False) -> Tensor:
    return 1. - paired_cosine_similarity(input1, input2, keepdim)


def pairwise_cosine_distance(input1: Tensor, input2: Tensor) -> Tensor:
    return 1. - pairwise_cosine_similarity(input1, input2)


def paired_distance(input1: Tensor, input2: Tensor, p: float = 2., keepdim: bool = False) -> Tensor:
    return F.pairwise_distance(input1, input2, p=p, eps=1e-06, keepdim=keepdim)


def pairwise_distance(input1: Tensor, input2: Tensor, p: float = 2.) -> Tensor:
    return F.pairwise_distance(
        torch.unsqueeze(input1, dim=1), torch.unsqueeze(input2, dim=0), p=p, eps=1e-6, keepdim=False)
