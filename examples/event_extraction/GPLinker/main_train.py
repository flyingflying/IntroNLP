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

from model import GPLinkerEEModel, GPLinkerEEConfig
from data_modules import argument_labels, event_labels, DuEEDataModule
from scheme import GPLinkerEEScheme
from metrics import GPLinkerEEMetrics, GPLinkerEEAnalysisMetrics


@dataclass
class HyperParameters:
    # 基础设置
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    output_dir = os.path.join(ROOT_DIR, "examples/event_extraction/GPLinker/output/")
    current_version: int = 0
    eval_interval: int = 10
    max_epochs: int = 200

    # 模型设置
    pretrained_name: str = "roberta"
    head_size: int = 64
    dropout: float = 0.3


class GPLinkerEESystem(LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(GPLinkerEESystem, self).__init__()
        self.save_hyperparameters(kwargs)

        config = GPLinkerEEConfig(
            n_argument_labels=len(argument_labels),
            head_size=self.hparams.head_size, pretrained_name=self.hparams.pretrained_name,
            dropout=self.hparams.dropout
        )
        self.model = GPLinkerEEModel(config)

        self.eval_flag = False
        self.metrics = GPLinkerEEMetrics(event_labels, scheme=GPLinkerEEScheme(argument_labels, ensure_trigger=True))
        self.analysis_metrics = GPLinkerEEAnalysisMetrics()

    def configure_optimizers(self) -> Any:
        # from torch.optim import Adam
        #
        # return Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_ratio * total_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp_metric": 0})  # noqa

    def on_train_epoch_start(self) -> None:
        self.eval_flag = True

    def training_step(self, batch: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        input_ids = batch["input_ids"]
        cal_mask = input_ids.ne(0).float()
        token_vectors = self.model.dropout(self.model.bert(input_ids, cal_mask)[0])

        # [batch_size, num_labels, num_tokens, num_tokens]
        arguments_logits = self.model.argument_classifier(token_vectors)
        heads_logits = self.model.head_classifier(token_vectors)
        tails_logits = self.model.tail_classifier(token_vectors)

        pair_mask = torch.triu(cal_mask.unsqueeze(1) * cal_mask.unsqueeze(-1))  # [batch_size, num_tokens, num_tokens]

        # [batch_size, num_tokens, num_tokens, num_tags+2]
        # all_logits = torch.cat([arguments_logits, heads_logits, tails_logits], dim=1).transpose(1, -1)
        # all_target = torch.cat([
        #     batch["arguments_tensor"].transpose(1, -1),
        #     batch["heads_tensor"].unsqueeze(-1),
        #     batch["tails_tensor"].unsqueeze(-1)
        # ], dim=-1)
        # loss = multi_label_cross_entropy_loss_with_mask(all_logits, all_target, pair_mask)

        arguments_loss = multi_label_cross_entropy_loss_with_mask(
            arguments_logits.permute(0, 2, 3, 1),  # 切记, 这里不是 transpose(1, -1)
            batch["arguments_tensor"].permute(0, 2, 3, 1),
            pair_mask
        )
        heads_loss = multi_label_cross_entropy_loss_with_mask(
            heads_logits.permute(0, 2, 3, 1),
            batch["heads_tensor"].unsqueeze(-1),
            pair_mask
        )
        tails_loss = multi_label_cross_entropy_loss_with_mask(
            tails_logits.permute(0, 2, 3, 1),
            batch["tails_tensor"].unsqueeze(-1),
            pair_mask
        )
        total_loss = (arguments_loss + heads_loss + tails_loss) / 3

        self.log("01_train_loss", total_loss)
        self.log("arguments_loss", arguments_loss)
        self.log("heads_loss", heads_loss)
        self.log("tails_loss", tails_loss)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.metrics = GPLinkerEEMetrics(event_labels, scheme=GPLinkerEEScheme(argument_labels, ensure_trigger=True))
        self.analysis_metrics = GPLinkerEEAnalysisMetrics()

    def on_test_epoch_start(self) -> None:
        self.eval_flag = True
        self.metrics = GPLinkerEEMetrics(event_labels, scheme=GPLinkerEEScheme(argument_labels, ensure_trigger=True))
        self.analysis_metrics = GPLinkerEEAnalysisMetrics()

    def on_validation_epoch_end(self) -> None:
        results = self.metrics.compute()

        self.log("hp_metric", results["micro"].f1_score)

        for idx, key in enumerate(["micro", "macro", "weighted_macro"]):
            result = results[key]
            self.log("{:02d}_{}_f1_score".format(idx+2, key), result.f1_score)
            self.log(f"{key}_precision", result.precision)
            self.log(f"{key}_recall", result.recall)

        analysis_results = self.analysis_metrics.compute()

        for key in ["arguments", "head", "tail"]:
            result = analysis_results[key]

            self.log(f"analysis_{key}_precision", result.precision)
            self.log(f"analysis_{key}_recall", result.recall)
            self.log(f"analysis_{key}_f1_score", result.f1_score)

    def on_test_epoch_end(self) -> None:
        self.print(self.metrics.classification_report())

        self.print(self.analysis_metrics.classification_report())

    def validation_step(self, batch: Dict[str, Tensor], *args, **kwargs):
        input_ids = batch["input_ids"]
        cal_mask = input_ids.ne(0).float()
        token_vectors = self.model.bert(input_ids, cal_mask)[0]

        arguments_logits = self.model.argument_classifier(token_vectors)
        heads_logits = self.model.head_classifier(token_vectors)
        tails_logits = self.model.tail_classifier(token_vectors)

        pair_mask = torch.triu(cal_mask.unsqueeze(1) * cal_mask.unsqueeze(-1))  # [batch_size, num_tokens, num_tokens]
        arguments_tensor = (arguments_logits > 0).float() * pair_mask.unsqueeze(1)
        heads_tensor = (heads_logits > 0).float().squeeze(1) * pair_mask
        tails_tensor = (tails_logits > 0).float().squeeze(1) * pair_mask

        if not self.eval_flag:
            arguments_tensor = torch.zeros_like(arguments_tensor, device=arguments_tensor.device)
            heads_tensor = torch.zeros_like(heads_tensor, device=heads_tensor.device)
            tails_tensor = torch.zeros_like(tails_tensor, device=tails_tensor.device)

        self.metrics.add_batch(
            references=batch["events"],  # noqa
            predictions=[arguments_tensor.detach().cpu(), heads_tensor.detach().cpu(), tails_tensor.detach().cpu()]
        )

        self.analysis_metrics.add_batch(
            references=[batch["arguments_tensor"], batch["heads_tensor"], batch["tails_tensor"]],
            predictions=[arguments_tensor, heads_tensor, tails_tensor]
        )

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
    system = GPLinkerEESystem(**asdict(hparams))
    datamodule = DuEEDataModule(hparams.batch_size, hparams.output_dir)

    trainer.fit(system, datamodule=datamodule)

    print("最佳模型分数", checkpoint.best_model_score)
    print("最佳模型路径", checkpoint.best_model_path)

    print("效果最好的测试结果: ")
    trainer.test(system, datamodule=datamodule, ckpt_path="best")

    # pytorch-lightning 似乎在这里有 bug, 需要进一步确认 !!!
    # print("最后一个模型的测试结果")  # 用于测试模型是欠拟合还是过拟合
    # trainer.test(
    #     system,
    #     datamodule=DuEEDataModule(hparams.batch_size, hparams.output_dir, test_over_fitting=True),
    #     ckpt_path="last"
    # )

    """
    hparams = HyperParameters()

    best_ckpt_name = "epoch=199-step=74600.ckpt"
    last_ckpt_name = "epoch=199-step=74600.ckpt"

    trainer = Trainer(accelerator="gpu", devices=[0, ])

    system = GPLinkerEESystem.load_from_checkpoint(
        os.path.join(hparams.output_dir, "checkpoint", last_ckpt_name)
    )
    datamodule = DuEEDataModule(hparams.batch_size, hparams.output_dir, test_over_fitting=True)

    trainer.test(system, datamodule=datamodule)
    """
