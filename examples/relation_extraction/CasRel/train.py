# Author: lqxu

import _prepare  # noqa

import os
from typing import *
from dataclasses import dataclass

from torch import Tensor
from core.trainer import loss_func
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from core.utils import ROOT_DIR
from core.utils import BasicMetrics
from model import CasRelModel, CasRelConfig

from data_modules import (
    relation_labels,
    init_hf_dataset,
    TrainCollateFn,
    test_collate_fn,
)


@dataclass
class HyperParameters:
    # 基础设置 (分词相关的设置和标签相关的设置见 data_modules 模块)
    batch_size: int = 32
    max_epochs: int = 10
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    learning_rate: float = 1e-5
    pretrained_model: str = "roberta"

    # 路径设置
    output_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/CasRel/output")

    # 其它设置
    current_version: int = 0
    use_modified_loss: bool = False  # 效果差不多, 可以不用考虑


class Metrics(BasicMetrics):
    def add(self, reference: Set[Tuple[int, ...]], prediction: Set[Tuple[int, ...]], **kwargs):
        for sro in reference:
            self.counters[sro[2]].gold_positive += 1
        for sro in prediction:
            self.counters[sro[2]].pred_positive += 1
            self.counters[sro[2]].true_positive += (1 if sro in reference else 0)

    def add_batch(self, references, predictions, **kwargs):
        for reference, prediction in zip(references, predictions):
            self.add(reference, prediction)


class CasRelSystem(LightningModule):
    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(CasRelSystem, self).__init__()
        self.save_hyperparameters(kwargs)

        # 初始化模型
        config = CasRelConfig(
            pretrained_name=self.hparams.pretrained_model,
            relation_labels=relation_labels
        )
        self.model = CasRelModel(config)

        # 其它设置
        self.hf_dataset_path = os.path.join(self.hparams.output_dir, "hf_dataset")
        self.hf_dataset = None
        self.num_labels = len(relation_labels)
        self.metrics = Metrics(labels=relation_labels)

    def prepare_data(self):
        if os.path.exists(self.hf_dataset_path):
            return
        init_hf_dataset(self.hf_dataset_path)

    def setup(self, stage: str):
        from datasets import DatasetDict
        self.hf_dataset = DatasetDict.load_from_disk(self.hf_dataset_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.hf_dataset["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=TrainCollateFn()
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        input_ids = batch["input_ids"]
        batch_size, num_tokens = input_ids.shape
        cal_mask = input_ids.ne(0).float()

        # step1: 对词语 ID 进行编码, 变成向量
        token_embeddings = self.model.bert.forward(
            input_ids=input_ids, attention_mask=cal_mask
        )[0]  # [batch_size, num_tokens, hidden_size]

        # step2: 计算 subject 部分的 logits 值
        subj_head_logits = self.model.subj_head_tagger(token_embeddings)  # [batch_size, num_tokens, 1]
        subj_tail_logits = self.model.subj_tail_tagger(token_embeddings)  # [batch_size, num_tokens, 1]

        # step3: 计算 subject 向量: subject 的头尾词向量取平均值
        """
        并行化的难度在这里: 每一个样本 subject 数量是不固定的。苏剑林大神给出的办法是每一个样本选择一个 subj - rel_obj 的对应关系去预测, 
        将 subject 的数量定死为 1。预测时也有这个问题, 苏剑林大神给出的办法是将 batch_size 设为 1。
        如果想彻底实现并行化, 还是比较麻烦的, 需要设置一个新的 mask, 同时也要考虑 subject 数过多时的情况, 以后可以尝试探索一下。
        这里使用的是 index 的策略选择 subject 的头尾词向量, 原代码使用的是 one-hot 策略。
        """
        batch_indices = [batch_idx for batch_idx in range(batch_size)]
        # 使用的是 "整型列表" 索引的方式, 用 "整型张量" 索引的方式也是可以的
        subj_head_embeddings = token_embeddings[batch_indices, batch["selected_subj_head"]]  # [batch_size, hidden_size]
        subj_tail_embeddings = token_embeddings[batch_indices, batch["selected_subj_tail"]]  # [batch_size, hidden_size]
        subj_embeddings = (subj_head_embeddings + subj_tail_embeddings) / 2  # [batch_size, hidden_size]

        # step4: 将 subject 向量融入 token_embeddings 中, 采用的是 "相加" 的方式
        token_embeddings = token_embeddings + subj_embeddings.unsqueeze(dim=1)  # [batch_size, num_tokens, hidden_size]

        # step5: 计算 rel_obj 部分的 logits 值
        rel_obj_head_logits = self.model.rel_obj_head_tagger(token_embeddings)  # [batch_size, num_tokens, num_labels]
        rel_obj_tail_logits = self.model.rel_obj_tail_tagger(token_embeddings)  # [batch_size, num_tokens, num_labels]

        # step6: 计算 loss 值
        if self.hparams.use_modified_loss:
            subj_head_loss = loss_func.multi_label_cross_entropy_loss_with_mask(
                subj_head_logits, batch["span_subj_heads"], cal_mask)
            subj_tail_loss = loss_func.multi_label_cross_entropy_loss_with_mask(
                subj_tail_logits, batch["span_subj_tails"], cal_mask)

            rel_obj_head_loss = loss_func.multi_label_cross_entropy_loss_with_mask(
                rel_obj_head_logits, batch["span_obj_heads"], cal_mask)
            rel_obj_tail_loss = loss_func.multi_label_cross_entropy_loss_with_mask(
                rel_obj_tail_logits, batch["span_obj_tails"], cal_mask)

            loss = (subj_head_loss + subj_tail_loss + rel_obj_head_loss + rel_obj_tail_loss) / 4
        else:
            subj_head_loss = loss_func.binary_cross_entropy_with_logits_and_mask(
                subj_head_logits, batch["span_subj_heads"], cal_mask)
            subj_tail_loss = loss_func.binary_cross_entropy_with_logits_and_mask(
                subj_tail_logits, batch["span_subj_tails"], cal_mask)
            rel_obj_head_loss = loss_func.binary_cross_entropy_with_logits_and_mask(
                rel_obj_head_logits, batch["span_obj_heads"], cal_mask)
            rel_obj_tail_loss = loss_func.binary_cross_entropy_with_logits_and_mask(
                rel_obj_tail_logits, batch["span_obj_tails"], cal_mask)
            # 一开始的 loss 非常大, 我觉得可以考虑除以 4
            loss = subj_head_loss + subj_tail_loss + rel_obj_head_loss + rel_obj_tail_loss
        self.log("01_train_loss", loss)
        return loss

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.hf_dataset["dev"],
            batch_size=1,  # CasRel 中的要求
            shuffle=False,
            num_workers=8,
            collate_fn=test_collate_fn
        )

    def on_validation_epoch_start(self):
        self.metrics = Metrics(relation_labels)

    def on_validation_epoch_end(self):
        results = self.metrics.compute(need_reset=True)
        self.log("02_dev_macro_f1", results["macro"].f1_score)
        self.log("03_dev_micro_f1", results["micro"].f1_score)

        for key in ["macro", "micro"]:
            self.log(f"{key}_dev_precision", results[key].precision)
            self.log(f"{key}_dev_recall", results[key].recall)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        input_ids, gold_sro_set = batch["input_ids"], batch["sro_set"]
        pred_sro_set = set(self.model.forward(input_ids))
        self.metrics.add(gold_sro_set, pred_sro_set)

    def on_test_epoch_start(self):
        self.metrics = Metrics(relation_labels)

    def on_test_epoch_end(self):
        self.print(self.metrics.classification_report(need_reset=True))

    # 没有测试集, 验证集当测试集用吧
    test_dataloader = val_dataloader
    test_step = validation_step

    def configure_optimizers(self):
        # 这里用 warmup and linear scheduler 效果不好, 因此就不用 scheduler 了
        from torch.optim import AdamW

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        return optimizer


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
        dirpath=checkpoint_dir, every_n_epochs=2, monitor="02_dev_macro_f1", save_top_k=2, mode="max")

    trainer = Trainer(
        accelerator="gpu", devices=[1, ],
        # accelerator="cpu",
        # 日志设置
        logger=logger, log_every_n_steps=50,
        # 回调函数设置
        callbacks=[checkpoint, ],
        # 其它设置
        max_epochs=hparams.max_epochs, amp_backend="native", check_val_every_n_epoch=2
    )

    system = CasRelSystem(**asdict(hparams))

    trainer.fit(system)

    print("最佳模型分数", checkpoint.best_model_score)
    print("最佳模型路径", checkpoint.best_model_path)

    trainer.test(system)
