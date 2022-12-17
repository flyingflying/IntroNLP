# Author: lqxu

import _prepare  # noqa

import os
from typing import *
from dataclasses import dataclass

import torch
from torch import Tensor
from pytorch_lightning import LightningModule

from core.utils import ROOT_DIR

from metrics import SROMetrics, AnalysisMetrics
from model import TPLinkerModel, TPLinkerConfig
from data_modules import DuIEDataModule, relation_labels, scheme


@dataclass
class HyperParameters:
    # 基础设置
    batch_size: int = 16
    learning_rate: float = 5e-5
    pretrained_model: str = "roberta"

    # epoch 和 step 相关的设置
    max_epochs: int = 20
    re_warmup_epochs: int = 2  # 用于 LR scheduler
    loss_weight_decay_steps: int = 10000  # loss_weight 衰减步数

    # 模型相关设置
    use_cln: bool = True

    # 路径相关设置
    output_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/TPLinker/output")

    # 其它设置
    current_version: int = 0
    eval_interval: int = 1


class TPLinkerSystem(LightningModule):
    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(TPLinkerSystem, self).__init__()
        self.save_hyperparameters(kwargs)
        self.num_relations = len(relation_labels)

        # ## 初始化模型
        config = TPLinkerConfig(
            pretrained_name=self.hparams.pretrained_model,
            num_relations=self.num_relations, use_cln=self.hparams.use_cln
        )
        self.model = TPLinkerModel(config)

        # ## 其它设置
        total_tasks = 2 * self.num_relations + 1
        self.entity_loss_weight = 1. / total_tasks
        self.tail_or_head_loss_weight = self.num_relations / total_tasks

        self.scheme = scheme
        self.eval_flag = False
        self.analysis_metrics, self.sro_metrics = AnalysisMetrics(), SROMetrics(relation_labels)

    def configure_optimizers(self):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)

        total_steps = self.trainer.estimated_stepping_batches
        t_0 = int(total_steps * (self.hparams.re_warmup_epochs / self.hparams.max_epochs))
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=1)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer, ], [scheduler, ]

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        input_ids = batch["input_ids"]
        entity_logits, head_logits, tail_logits = self.model(input_ids)

        entity_loss = torch.nn.functional.cross_entropy(entity_logits, batch["entity_tensor"])
        head_loss = torch.nn.functional.cross_entropy(head_logits, batch["head_tensor"])
        tail_loss = torch.nn.functional.cross_entropy(tail_logits, batch["tail_tensor"])

        loss = self.entity_loss_weight * entity_loss + self.tail_or_head_loss_weight * (head_loss + tail_loss)
        self.log("1_train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.analysis_metrics = AnalysisMetrics()
        self.sro_metrics = SROMetrics(relation_labels)

    def on_train_epoch_start(self): self.eval_flag = True

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):

        input_ids = batch["input_ids"]
        entity_logits, head_logits, tail_logits = self.model(input_ids)

        entity_tensor = entity_logits.argmax(dim=1)
        head_tensor = head_logits.argmax(dim=1)
        tail_tensor = tail_logits.argmax(dim=1)

        entity_tensor[batch["ignored_mask"]] = 0
        head_tensor[batch["ignored_mask"]] = 0
        tail_tensor[batch["ignored_mask"]] = 0

        self.analysis_metrics.add_batch(references=batch["entity_tensor"], predictions=entity_tensor, name="entity")
        self.analysis_metrics.add_batch(references=batch["head_tensor"], predictions=head_tensor, name="head")
        self.analysis_metrics.add_batch(references=batch["tail_tensor"], predictions=tail_tensor, name="tail")

        for idx, sro_set in enumerate(batch["sro_sets"]):
            if self.eval_flag:  # 一定不要在一开始就进行测试, 建议训练两轮后再开始测试, 不然会由于解码量太大而卡住
                prediction = scheme.decode(entity_tensor[idx], head_tensor[idx], tail_tensor[idx])  # noqa
            else:
                prediction = set()
            self.sro_metrics.add(
                reference=sro_set,
                prediction=prediction
            )

    def on_validation_epoch_end(self):
        results = self.sro_metrics.compute()
        order = {"macro": 2, "micro": 3, "weighted_macro": 4}

        for key in results.keys():
            if key not in order.keys():  # 只记录平均量
                continue
            o = order[key]
            self.log(f"{o}_{key}_f1_score", results[key].f1_score)
            self.log(f"{key}_precision", results[key].precision)
            self.log(f"{key}_recall", results[key].recall)

        results = self.analysis_metrics.compute()
        for key in results.keys():
            if key in ["macro", "micro", "weighted_macro"]:  # 不记录平均量
                continue
            self.log(f"{key}_f1_score", results[key].f1_score)
            self.log(f"{key}_precision", results[key].precision)
            self.log(f"{key}_recall", results[key].recall)

    def on_test_epoch_start(self):
        self.eval_flag = True
        self.analysis_metrics = AnalysisMetrics()
        self.sro_metrics = SROMetrics(relation_labels)

    test_step = validation_step

    def on_test_epoch_end(self):
        self.print(self.analysis_metrics.classification_report())
        self.print(self.sro_metrics.classification_report())


if __name__ == '__main__':

    import shutil
    from dataclasses import asdict
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    hparams = HyperParameters()

    # 设置 tensorboard 的 logger
    logger_dir = os.path.join(hparams.output_dir, "lightning_logs")
    logger_version_dir = os.path.join(logger_dir, f"version_{hparams.current_version}")
    if os.path.exists(logger_version_dir):
        shutil.rmtree(logger_version_dir)
    logger = TensorBoardLogger(save_dir=hparams.output_dir, name="lightning_logs", version=hparams.current_version)

    # 设置 checkpoint
    checkpoint_dir = os.path.join(hparams.output_dir, "checkpoint")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, every_n_epochs=hparams.eval_interval,
        monitor="3_micro_f1_score", save_top_k=2, mode="max"
    )

    trainer = Trainer(
        accelerator="gpu", devices=[0, ],
        # 日志设置
        logger=logger, log_every_n_steps=50,
        # 回调函数设置
        callbacks=[checkpoint, ],
        # 其它设置
        max_epochs=hparams.max_epochs, amp_backend="native", check_val_every_n_epoch=hparams.eval_interval
    )
    system = TPLinkerSystem(**asdict(hparams))
    datamodule = DuIEDataModule(hparams.batch_size)

    trainer.fit(system, datamodule=datamodule)

    print("最佳模型分数", checkpoint.best_model_score)
    print("最佳模型路径", checkpoint.best_model_path)

    trainer.test(system, datamodule=datamodule, ckpt_path="best")

    """
    from pytorch_lightning import Trainer

    hparams = HyperParameters()

    ckpt_path = os.path.join(hparams.output_dir, "checkpoint", "epoch=11-step=124728.ckpt")

    system = TPLinkerSystem.load_from_checkpoint(ckpt_path)

    trainer = Trainer(accelerator="gpu", devices=[0, ])

    trainer.test(system, datamodule=DuIEDataModule(hparams.batch_size))
    """
