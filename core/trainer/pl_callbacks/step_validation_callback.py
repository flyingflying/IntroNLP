# Author: lqxu

"""
在 PyTorch-Lightning 1.8.2 中, 如果在 `ModelCheckpoint` 中使用 `every_n_train_steps`,
并且在 `Trainer` 中使用 `val_check_interval`, 此时会先保存 model checkpoint, 再进行 validation。
这实在是不能接受, 于是参考: https://github.com/Lightning-AI/lightning/issues/2534 有了这个 callback 。

如果程序出现 **could not find the monitored key in the returned metrics** 的警告, 你就要考虑这个模块了。
"""

from typing import *

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import RunningStage


__all__ = ["StepValidationCallback"]


class StepValidationCallback(pl.Callback):
    def __init__(self, every_n_steps):
        self.last_run = None
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        # Prevent Running validation many times in gradient accumulation
        if trainer.global_step == self.last_run:
            return
        else:
            self.last_run = None
        if trainer.global_step % self.every_n_steps == 0 and trainer.global_step != 0:
            trainer.training = False
            stage = trainer.state.stage
            trainer.state.stage = RunningStage.VALIDATING
            trainer._run_evaluate()  # noqa
            trainer.state.stage = stage
            trainer.training = True
            trainer._logger_connector._epoch_end_reached = False  # noqa
            self.last_run = trainer.global_step
