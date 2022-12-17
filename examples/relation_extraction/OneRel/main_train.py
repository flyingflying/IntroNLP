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
from model import OneRelModel, OneRelConfig
from data_modules import DuIEDataModule, relation_labels, scheme


@dataclass
class HyperParameters:
    # 基础设置
    batch_size: int = 12
    learning_rate: float = 1e-5
    pretrained_model: str = "roberta"

    # epoch 和 step 相关的设置
    max_epochs: int = 20

    # 模型相关设置
    dropout1: float = 0.2  # 用于对 token_embeddings 进行 dropout
    dropout2: float = 0.1  # 用于对 token_pairs_embeddings 进行 dropout

    # 路径相关设置
    output_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/OneRel/output")

    # 其它设置
    current_version: int = 0
    eval_interval: int = 1


class OneRelSystem(LightningModule):
    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(OneRelSystem, self).__init__()
        self.save_hyperparameters(kwargs)
        self.num_relations = len(relation_labels)

        # ## 初始化模型
        config = OneRelConfig(
            pretrained_name=self.hparams.pretrained_model,
            num_relations=self.num_relations, dropout1=self.hparams.dropout1, dropout2=self.hparams.dropout2
        )
        self.model = OneRelModel(config)

        # ## 其它设置
        self.scheme = scheme
        self.eval_flag = False
        self.analysis_metrics, self.sro_metrics = AnalysisMetrics(), SROMetrics(relation_labels)

    def configure_optimizers(self):
        from torch.optim import Adam
        optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        input_ids = batch["input_ids"]
        logits = self.model(input_ids)  # [batch_size, 4, num_relations, num_tokens, num_tokens]
        loss = torch.nn.functional.cross_entropy(logits, batch["target"])

        self.log("1_train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.analysis_metrics = AnalysisMetrics()
        self.sro_metrics = SROMetrics(relation_labels)

    def on_train_epoch_start(self): self.eval_flag = True

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):

        input_ids = batch["input_ids"]
        logits = self.model(input_ids)

        result = logits.argmax(dim=1)  # [batch_size, num_relations, num_subjects, num_objects]

        result = result.permute(0, 2, 3, 1)
        result[batch["ignored_mask"]] = 0
        result = result.permute(0, 3, 1, 2)

        self.analysis_metrics.add_batch(references=batch["target"], predictions=result)

        for idx, sro_set in enumerate(batch["sro_sets"]):
            if self.eval_flag:  # 一定不要在一开始就进行测试, 建议训练两轮后再开始测试, 不然会由于解码量太大而卡住
                prediction = scheme.decode(result[idx])
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
    system = OneRelSystem(**asdict(hparams))
    datamodule = DuIEDataModule(hparams.batch_size)

    trainer.fit(system, datamodule=datamodule)

    print("最佳模型分数", checkpoint.best_model_score)
    print("最佳模型路径", checkpoint.best_model_path)

    trainer.test(system, datamodule=datamodule, ckpt_path="best")
