# Author: lqxu

import _prepare  # noqa

import os
from typing import *
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa
from pytorch_lightning import LightningModule

from core.utils import ROOT_DIR
from core.utils import BasicMetrics
from model import PRGCModel, PRGCConfig
from data_modules import relation_labels

_SRO_TYPING = Tuple[int, int, int, int, int]


class SROMetrics(BasicMetrics):

    """ 计算 SRO 的 metrics """

    def add(self, reference: Set[_SRO_TYPING], prediction: Set[_SRO_TYPING], **kwargs):
        for sro in reference:
            self.counters[sro[2]].gold_positive += 1
        for sro in prediction:
            self.counters[sro[2]].pred_positive += 1
            self.counters[sro[2]].true_positive += (1 if sro in reference else 0)

    def add_batch(self, references: List[Any], predictions: List[Any], **kwargs):
        for reference, prediction in zip(references, predictions):
            self.add(reference, prediction)


@dataclass
class HyperParameters:
    # 基础设置
    pretrained_model: str = "roberta"
    batch_size: int = 20
    max_epochs: int = 10

    # 优化器相关
    bert_learning_rate: float = 5e-5
    downstream_learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0

    # 模型相关
    classifier_dropout: float = 0.1

    # 路径相关设置
    output_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/PRGC/output")

    # 其它设置
    current_version: int = 0
    eval_interval: int = 1


class PRGCSystem(LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(PRGCSystem, self).__init__()
        self.save_hyperparameters(kwargs)

        # 初始化模型
        config = PRGCConfig(
            num_relations=len(relation_labels), classifier_dropout=self.hparams.classifier_dropout)
        self.model = PRGCModel(config)

        self.val_flag = False
        self.metrics = SROMetrics(relation_labels)

    def configure_optimizers(self):
        from torch.optim import Adam
        from transformers.optimization import get_linear_schedule_with_warmup

        # 将模型参数分成四组, 对应不同的 weight decay 和 learning rate
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_names = {f"bert.{n}" for n, p in self.model.bert.named_parameters()}
        downstream_params = [(n, p) for n, p in self.model.named_parameters() if n not in bert_param_names]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay, "lr": self.hparams.bert_learning_rate
            },
            {
                "params": [p for n, p in self.model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0, "lr": self.hparams.bert_learning_rate
            },
            {
                "params": [p for n, p in downstream_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay, "lr": self.hparams.downstream_learning_rate
            },
            {
                "params": [p for n, p in downstream_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0, "lr": self.hparams.downstream_learning_rate
            }
        ]
        optimizer = Adam(optimizer_grouped_parameters)

        # 根据步数设立 lr scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        # ## step1: 编码词向量
        input_ids = batch["input_ids"]
        batch_size, num_tokens = input_ids.shape
        cal_mask = input_ids.ne(0).float()  # [batch_size, num_tokens]
        token_vectors = self.model.bert(input_ids, cal_mask)[0]  # [batch_size, num_tokens, hidden_size]

        # ## step2: 生成句向量
        seq_cal_mask = cal_mask.unsqueeze(-1)  # [batch_size, num_tokens, 1]
        sentence_vector = torch.sum(token_vectors * seq_cal_mask, dim=1) / torch.sum(seq_cal_mask, dim=1)

        # ## step3: 预测关系, 并计算 loss
        relation_logits = self.model.relation_judgement(sentence_vector)  # [batch_size, num_relations]
        relation_loss = F.binary_cross_entropy_with_logits(relation_logits, batch["relations"], reduction="mean")

        # ## step4: 预测 correspondence, 并计算 loss
        correspondence_logits = self.model.global_correspondence(
            torch.cat([
                token_vectors.unsqueeze(2).expand(-1, -1, num_tokens, -1),
                token_vectors.unsqueeze(1).expand(-1, num_tokens, -1, -1)
            ], dim=-1)
        ).squeeze(dim=-1)  # [batch_size, num_tokens, num_tokens]

        correspondence_loss = F.binary_cross_entropy_with_logits(
            correspondence_logits, batch["correspondence"], reduction="none"
        )  # [batch_size, num_tokens, num_tokens]

        pair_cal_mask = cal_mask.unsqueeze(1) * cal_mask.unsqueeze(-1)  # [batch_size, num_tokens, num_tokens]
        correspondence_loss = torch.sum(correspondence_loss * pair_cal_mask) / torch.sum(pair_cal_mask)

        # ## step5: 预测 subjects 和 objects
        relation_vectors = self.model.relation_embeddings(batch["selected_relations"])  # [batch_size, hidden_size]
        new_token_vectors = token_vectors + relation_vectors.unsqueeze(1)  # [batch_size, num_tokens, hidden_size]
        new_token_vectors = self.model.dense(new_token_vectors)
        subject_logits = self.model.subject_classifier(new_token_vectors).transpose(1, 2)  # [batch_size, 3, num_tokens]
        subject_loss = F.cross_entropy(subject_logits, batch["subjects"], reduction="none")  # [batch_size, num_tokens]
        subject_loss = torch.sum(subject_loss * cal_mask) / torch.sum(cal_mask)
        object_logits = self.model.object_classifier(new_token_vectors).transpose(1, 2)  # [batch_size, 3, num_tokens]
        object_loss = F.cross_entropy(object_logits, batch["objects"], reduction="none")  # [batch_size, num_tokens]
        object_loss = torch.sum(object_loss * cal_mask) / torch.sum(cal_mask)

        # ## step6: 综合 loss
        total_loss = relation_loss + correspondence_loss + (subject_loss + object_loss) / 2

        self.log("relation_loss", relation_loss)
        self.log("correspondence_loss", correspondence_loss)
        self.log("subject_loss", subject_loss)
        self.log("object_loss", object_loss)
        self.log("01_train_loss", total_loss)
        return total_loss

    def on_train_epoch_start(self):
        self.val_flag = True

    def on_validation_epoch_start(self):
        self.metrics = SROMetrics(relation_labels)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if not self.val_flag:
            return

        pred_sro_sets = self.model.decode(batch["input_ids"])

        gold_sro_sets = batch["sro_sets"]

        self.metrics.add_batch(gold_sro_sets, pred_sro_sets)

    def on_validation_epoch_end(self):
        if not self.val_flag:
            return

        results = self.metrics.compute()

        self.log("02_macro_f1_score", results["macro"].f1_score)
        self.log("03_micro_f1_score", results["micro"].f1_score)
        self.log("04_weighted_macro_f1_score", results["weighted_macro"].f1_score)

        for key in ["macro", "micro", "weighted_macro"]:
            self.log(f"{key}_precision", results[key].precision)
            self.log(f"{key}_recall", results[key].recall)

    on_test_epoch_start = on_validation_epoch_start
    test_step = validation_step

    def on_test_epoch_end(self) -> None:
        self.print(self.metrics.classification_report())


if __name__ == '__main__':

    import shutil
    from dataclasses import asdict

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from data_modules import DuIE2DataModule

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
        monitor="03_micro_f1_score", save_top_k=2, mode="max"
    )

    trainer = Trainer(
        accelerator="gpu", devices=[0, ],
        # 日志设置
        logger=logger, log_every_n_steps=50,
        # 回调函数设置
        callbacks=[checkpoint, ],
        # 其它设置
        max_epochs=hparams.max_epochs, amp_backend="native", check_val_every_n_epoch=hparams.eval_interval,
        # 梯度裁剪
        gradient_clip_algorithm="norm", gradient_clip_val=2.,
    )
    system = PRGCSystem(**asdict(hparams))
    datamodule = DuIE2DataModule(hparams.batch_size)

    trainer.fit(system, datamodule=datamodule)

    print("最佳模型分数", checkpoint.best_model_score)
    print("最佳模型路径", checkpoint.best_model_path)

    trainer.test(system, datamodule=datamodule, ckpt_path="best")
