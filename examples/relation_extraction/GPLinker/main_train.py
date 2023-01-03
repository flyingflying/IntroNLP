# Author: lqxu

import _prepare  # noqa

import os
from typing import *
from dataclasses import dataclass, asdict

import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer

from core.utils import ROOT_DIR
from core.trainer.loss_func import multi_label_cross_entropy_loss_with_mask

from model import GPLinkerModel, GPLinkerConfig
from metrics import AnalysisMetrics, SROMetrics
from data_modules import DuIEDataModule, relation_labels, scheme


@dataclass
class HyperParameters:
    # 基础设置
    batch_size: int = 32
    learning_rate: float = 2e-5
    output_dir = os.path.join(ROOT_DIR, "examples/relation_extraction/GPLinker/output")

    # 训练设置
    eval_interval: int = 1
    max_epochs: int = 15

    # 模型设置
    pretrained_model: str = "roberta"

    # 其它设置
    current_version: int = 0


class GPLinkerRESystem(LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(GPLinkerRESystem, self).__init__()

        self.save_hyperparameters(kwargs)

        config = GPLinkerConfig(
            pretrained_name=self.hparams.pretrained_model, num_relations=len(relation_labels)
        )
        self.model = GPLinkerModel(config)

        self.scheme = scheme
        self.eval_flag = False
        self.analysis_metrics, self.sro_metrics = AnalysisMetrics(), SROMetrics(relation_labels)

    def configure_optimizers(self):
        from torch.optim import Adam

        return Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp_metric": 0})  # noqa

    def on_train_epoch_start(self) -> None:
        self.eval_flag = True

    def training_step(self, batch: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        input_ids = batch["input_ids"]
        cal_mask = input_ids.ne(0).float()
        token_vectors = self.model.bert(input_ids, cal_mask)[0]

        entity_logits = self.model.entity_classifier(token_vectors)  # [batch_size, 2, n_tokens, n_tokens]
        head_logits = self.model.head_classifier(token_vectors)  # [batch_size, n_relations, n_tokens, n_tokens]
        tail_logits = self.model.tail_classifier(token_vectors)  # [batch_size, n_relations, n_tokens, n_tokens]

        pair_mask = cal_mask.unsqueeze(1) * cal_mask.unsqueeze(-1)  # [batch_size, num_tokens, num_tokens]

        entity_loss = multi_label_cross_entropy_loss_with_mask(
            entity_logits.permute(0, 2, 3, 1),  # 切记, 这里不是 transpose(1, -1)
            batch["entity_tensor"].permute(0, 2, 3, 1),
            torch.triu(pair_mask)  # 实体识别需要 mask 掉下三角
        )
        heads_loss = multi_label_cross_entropy_loss_with_mask(
            head_logits.permute(0, 2, 3, 1),
            batch["head_tensor"].permute(0, 2, 3, 1),
            pair_mask
        )
        tails_loss = multi_label_cross_entropy_loss_with_mask(
            tail_logits.permute(0, 2, 3, 1),
            batch["tail_tensor"].permute(0, 2, 3, 1),
            pair_mask
        )
        total_loss = (entity_loss + heads_loss + tails_loss) / 3

        self.log("01_train_loss", total_loss)
        self.log("entity_loss", entity_loss)
        self.log("heads_loss", heads_loss)
        self.log("tails_loss", tails_loss)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.sro_metrics = SROMetrics(relation_labels, scheme=self.scheme)
        self.analysis_metrics = AnalysisMetrics()

    def on_test_epoch_start(self) -> None:
        self.eval_flag = True
        self.sro_metrics = SROMetrics(relation_labels, scheme=self.scheme)
        self.analysis_metrics = AnalysisMetrics()

    def on_validation_epoch_end(self) -> None:
        results = self.sro_metrics.compute()

        self.log("hp_metric", results["micro"].f1_score)

        for idx, key in enumerate(["micro", "macro", "weighted_macro"]):
            result = results[key]
            self.log("{:02d}_{}_f1_score".format(idx+2, key), result.f1_score)
            self.log(f"{key}_precision", result.precision)
            self.log(f"{key}_recall", result.recall)

        analysis_results = self.analysis_metrics.compute()

        for key in self.analysis_metrics.labels:
            result = analysis_results[key]

            self.log(f"analysis_{key}_precision", result.precision)
            self.log(f"analysis_{key}_recall", result.recall)
            self.log(f"analysis_{key}_f1_score", result.f1_score)

    def on_test_epoch_end(self) -> None:
        self.print(self.sro_metrics.classification_report())

        self.print(self.analysis_metrics.classification_report())

    def validation_step(self, batch: Dict[str, Tensor], *args, **kwargs):
        input_ids = batch["input_ids"]
        cal_mask = input_ids.ne(0).float()
        token_vectors = self.model.bert(input_ids, cal_mask)[0]

        entity_logits = self.model.entity_classifier(token_vectors)  # [batch_size, 2, n_tokens, n_tokens]
        head_logits = self.model.head_classifier(token_vectors)  # [batch_size, n_relations, n_tokens, n_tokens]
        tail_logits = self.model.tail_classifier(token_vectors)  # [batch_size, n_relations, n_tokens, n_tokens]

        pair_mask = cal_mask.unsqueeze(1) * cal_mask.unsqueeze(-1)  # [batch_size, num_tokens, num_tokens]
        pair_mask = pair_mask.unsqueeze(1)  # [batch_size, 1, num_tokens, num_tokens]
        entity_tensor = (entity_logits > 0).float() * torch.triu(pair_mask)
        head_tensor = (head_logits > 0).float() * pair_mask
        tail_tensor = (tail_logits > 0).float() * pair_mask

        if not self.eval_flag:
            entity_tensor = torch.zeros_like(entity_tensor, device=entity_tensor.device)
            head_tensor = torch.zeros_like(head_tensor, device=head_tensor.device)
            tail_tensor = torch.zeros_like(tail_tensor, device=tail_tensor.device)

        for idx, sro_set in enumerate(batch["sro_sets"]):
            prediction = self.scheme.decode(
                entity_tensor[idx, 0], entity_tensor[idx, 1], head_tensor[idx], tail_tensor[idx])
            self.sro_metrics.add(
                reference=sro_set,  # noqa
                prediction=prediction
            )

        self.analysis_metrics.add_batch(references=batch["entity_tensor"], predictions=entity_tensor, name="entity")
        self.analysis_metrics.add_batch(references=batch["head_tensor"], predictions=head_tensor, name="head")
        self.analysis_metrics.add_batch(references=batch["tail_tensor"], predictions=tail_tensor, name="tail")

    test_step = validation_step


if __name__ == '__main__':

    import shutil
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    hparams = HyperParameters()

    # 设置 tensorboard 的 logger
    logger_dir = os.path.join(hparams.output_dir, "lightning_logs")
    logger_version_dir = os.path.join(logger_dir, f"version_{hparams.current_version}")
    if os.path.exists(logger_version_dir):
        shutil.rmtree(logger_version_dir)
    logger = TensorBoardLogger(
        save_dir=hparams.output_dir, name="lightning_logs", version=hparams.current_version,
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-hyperparameters
        default_hp_metric=False
    )

    # 设置 checkpoint
    checkpoint_dir = os.path.join(hparams.output_dir, "checkpoint")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, every_n_epochs=hparams.eval_interval,
        monitor="02_micro_f1_score", save_top_k=2, mode="max", save_last=True
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
    system = GPLinkerRESystem(**asdict(hparams))
    datamodule = DuIEDataModule(hparams.batch_size)

    trainer.fit(system, datamodule=datamodule)

    print("最佳模型分数", checkpoint.best_model_score)
    print("最佳模型路径", checkpoint.best_model_path)

    print("效果最好的测试结果: ")
    trainer.test(system, datamodule=datamodule, ckpt_path="best")

    """
    hparams = HyperParameters()

    system = GPLinkerRESystem.load_from_checkpoint(
        os.path.join(hparams.output_dir, "checkpoint", "epoch=12-step=67561.ckpt")
    )

    datamodule = DuIEDataModule(hparams.batch_size)

    trainer = Trainer(accelerator="gpu", devices=[0, ])

    trainer.test(system, datamodule)
    """
