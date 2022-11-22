# Author: lqxu

""" 对于 Python 的 typing 功能的支持 """

from typing import *
from abc import ABC

from torch.optim import Optimizer

__all__ = ["LRScheduler", ]


class LRScheduler(ABC):
    """ PyCharm (2021.3.1) 对于 pyi 文件中的 protected member 引用会报错, 因此这里单独拿出来, 用作 lr_scheduler 的 typing """

    optimizer: Optimizer
    base_lrs: List[float]
    last_epoch: int
    verbose: bool

    # noinspection PyUnusedLocal
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False): raise NotImplementedError
    def state_dict(self) -> dict: raise NotImplementedError
    def load_state_dict(self, state_dict: dict): raise NotImplementedError
    def get_last_lr(self) -> List[float]: raise NotImplementedError
    def get_lr(self) -> float: raise NotImplementedError
    def step(self, epoch: int = None): raise NotImplementedError
    def print_lr(self, is_verbose: bool, group: dict, lr: float, epoch: int = None): raise NotImplementedError
