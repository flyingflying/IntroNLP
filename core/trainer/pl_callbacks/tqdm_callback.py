# Author: lqxu

"""
在 PyTorch-Lightning 1.8.2 中, validation 的步数也会算到 total 的步数当中, 这个模块就是为了解决这个问题

我自己写的, 没有参考 web 资料, 如果有问题, 非常欢迎提 issue
"""

from typing import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n  # noqa

__all__ = ["TQDMCallback", ]


class TQDMCallback(TQDMProgressBar):
    def on_validation_batch_end(self, trainer: "pl.Trainer", *_: Any) -> None:
        """ 删除更新主 progress bar """
        if self._should_update(self.val_batch_idx, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, self.val_batch_idx)

    @property
    def total_val_batches(self) -> Union[int, float]:
        """ 直接让其为 0 就好啦 """
        return 0

    @property
    def _val_processed(self) -> int:
        """ 直接让其为 0 就好啦 """
        return 0
