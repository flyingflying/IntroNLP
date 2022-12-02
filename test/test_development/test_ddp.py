# Author: lqxu

"""  """

from typing import *

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class TestDDP(pl.LightningModule):
    def __init__(self):
        super(TestDDP, self).__init__()

        self.model = torch.nn.Linear(in_features=100, out_features=10)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=self.model.parameters(), lr=1e-5
        )

    def train_dataloader(self) -> DataLoader:
        dataset = [
            [1., ] * 100, [2., ] * 100
        ]

        def collate_fn(batch):
            return {
                "input_tensor": torch.tensor(batch)
            }

        return DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)  # noqa

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self.model(batch["input_tensor"])

        """
        在 PyTorch 中, all_gather 是没有梯度的, 如果需要梯度, 可以参考: SimCLR 的代码:
            https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
        PyTorch-Lightning 中已经帮助我们处理好啦, 不需要操心这个问题了
        """

        gathered_logits = self.all_gather(logits, sync_grads=True)
        # print(logits)
        # print("local rank:", self.local_rank)
        # print("global rank: ", self.global_rank)
        assert (torch.all(gathered_logits[self.global_rank] == logits)).item()  # noqa
        return logits.sum()


if __name__ == '__main__':
    # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html#distributed-data-parallel
    # trainer = pl.Trainer(
    #     logger=False, enable_checkpointing=False, max_epochs=100, strategy="ddp", accelerator="gpu", devices=[0, 1])

    # trainer = pl.Trainer(
    #     logger=False, enable_checkpointing=False, max_steps=1, accelerator="gpu", devices=[0, ])

    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False, max_epochs=3, accelerator="cpu")

    trainer.fit(TestDDP())
