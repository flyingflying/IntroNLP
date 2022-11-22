# Author: lqxu

import warnings

import torch
from torch import Tensor

__all__ = ["CheckWarning", "is_same_tensor"]


class CheckWarning(Warning):
    """ check 模块的 warning 类 """


warnings.filterwarnings(action="always", category=CheckWarning)


def is_same_tensor(input1: Tensor, input2: Tensor, eps: float = 1e-6) -> bool:
    if input1.dtype != input2.dtype:
        warnings.warn("两个张量的类型不一致", CheckWarning)
        return False
    if input1.shape != input2.shape:
        warnings.warn("两个张量的 shape 不一致", CheckWarning)
        return False
    result = torch.abs(input1 - input2) < eps
    if not torch.all(result).item():
        ratio = 1 - torch.sum(result) / torch.numel(input1)
        ratio = round(ratio.item(), 5) * 100
        warnings.warn(f"两个张量的数值有 {ratio}% 的数值不符合要求")
        return False
    return True
