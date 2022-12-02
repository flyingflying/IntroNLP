# Author: lqxu

"""
paper: SimCSE: Simple Contrastive Learning of Sentence Embeddings \n
    link: https://aclanthology.org/2021.emnlp-main.552.pdf \n
    code: https://github.com/princeton-nlp/SimCSE \n

有监督的 SimCSE, 使用 OCNLI 数据集进行训练, LCQMC / CNSD-STS-B 数据集进行验证 \n

LCQMC: test_spearmanr=71.76; eval_spearmanr=68.41
CNSD-STS-B: test_spearmanr=72.16; eval_spearmanr=77.30

hard_negative_logits 参数不要设置, 效果会越来越差

运行方式: python examples/sentence_embedding/02_s_sim_cse.py

主要的依赖库: pytorch==1.13.0, transformers==4.24.0, pytorch-lightning==1.8.2
"""

import _prepare  # noqa

import os
from typing import *
from dataclasses import dataclass, asdict

import torch
import pandas as pd
from torch import nn
from torch import Tensor
import pytorch_lightning as pl
# noinspection PyPep8Naming
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core import utils
from core.utils import vector_pair
from core.models import SentenceBertModel, SentenceBertConfig


@dataclass
class HyperParameters:
    train_batch_size: int = 128
    eval_or_test_batch_size: int = 512
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 模型相关
    bert_dropout: float = 0.1
    use_mean_pooling: bool = False
    use_max_pooling: bool = False
    use_first_token_pooling: bool = True
    pooling_with_mlp: bool = True
    temperature: float = 0.05
    add_extra_mlp: bool = False  # 对应原代码里面的 mlp_only_train 参数, 即在训练阶段有 MLP, 在测试阶段没有 MLP
    hard_negative_logits: float = 0.  # 对应原代码中的 hard_negative_weight 参数, 应该是正数, 对训练提出了更高的要求 (效果不好)

    # 分词器相关
    max_sequence_length: int = 64


def prepare_train_data() -> pd.DataFrame:
    data_dir = os.path.join(utils.DATA_DIR, "sentence_embeddings/OCNLI")
    file_names = ["train.50k.jsonl", "dev.jsonl"]

    dfs = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)

        part_df = pd.read_json(file_path, lines=True)
        part_df = part_df.set_index("id")
        part_df = part_df[["sentence1", "sentence2", "label"]]
        part_df = part_df[part_df["label"] != "neutral"]  # 不需要中立的数据

        dfs.append(part_df)

    total_df = pd.concat(dfs, axis="index", ignore_index=True)

    anchors, positives, negatives = [], [], []
    for anchor, group_df in total_df.groupby("sentence1"):
        positive, negative = None, None
        for _, row in group_df.iterrows():
            if row["label"] == "entailment":
                positive = row["sentence2"]
            if row["label"] == "contradiction":
                negative = row["sentence2"]
        if positive is not None and negative is not None:
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

    return pd.DataFrame({
        "anchor": anchors, "positive": positives, "negative": negatives
    })


class SupervisedSimCSE(pl.LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        """ 初始化模型 """

        super(SupervisedSimCSE, self).__init__()
        # 保存配置参数
        self.save_hyperparameters(kwargs)

        # 根据参数生成模型
        config = SentenceBertConfig(
            pretrained_name="roberta",
            use_mean_pooling=self.hparams.use_mean_pooling,
            pooling_with_mlp=self.hparams.pooling_with_mlp,
            use_max_pooling=self.hparams.use_max_pooling,
            use_first_token_pooling=self.hparams.use_first_token_pooling
        )
        bert_kwargs = {
            "hidden_dropout_prob": self.hparams.bert_dropout,
            "attention_probs_dropout_prob": self.hparams.bert_dropout,
        }
        self.model = SentenceBertModel(config, bert_kwargs)

        if self.hparams.add_extra_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(config.bert_config.hidden_size, config.bert_config.hidden_size), nn.Tanh())
        else:
            self.mlp = None

        # 分词器
        self.tokenizer = utils.get_default_tokenizer()
        self.tokenize_kwargs = {
            "max_length": self.hparams.max_sequence_length,
            "padding": "max_length", "truncation": True,
            "return_attention_mask": False, "return_token_type_ids": False
        }

    @staticmethod
    def get_grouped_parameters(model, weight_decay, learning_rate):

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": learning_rate
            },
        ]

        return optimizer_grouped_parameters

    def configure_optimizers(self):
        """
        采用 transformers 默认的 optimizer 和 lr_scheduler \n
        reference:
            https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
        """

        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

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

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_ratio * total_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        df = prepare_train_data()
        df["anchor_input_ids"] = self.tokenizer(df["anchor"].to_list(), **self.tokenize_kwargs)["input_ids"]
        df["positive_input_ids"] = self.tokenizer(df["positive"].to_list(), **self.tokenize_kwargs)["input_ids"]
        df["negative_input_ids"] = self.tokenizer(df["negative"].to_list(), **self.tokenize_kwargs)["input_ids"]
        df.drop(columns=["anchor", "positive", "negative"])

        return DataLoader(
            dataset=utils.DataFrameDataset(df),
            collate_fn=utils.DictDataCollator(
                text_keys=["anchor_input_ids", "positive_input_ids", "negative_input_ids"]),
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=8,
            # ** 由于使用的是 in-batch negatives 进行训练, 这里最好是直接将无法满足 batch_size 的 mini-batch 直接抛弃掉
            drop_last=True
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        anchor = batch["anchor_input_ids"]
        positive, negative = batch["positive_input_ids"], batch["negative_input_ids"]
        batch_size = anchor.size(0)
        anchor_vector = self.model(anchor)["pooler_output"]  # [batch_size, hidden_size]
        positive_vector = self.model(positive)["pooler_output"]  # [batch_size, hidden_size]
        negative_vector = self.model(negative)["pooler_output"]  # [batch_size, hidden_size]

        if self.mlp is not None:
            anchor_vector = self.mlp(anchor_vector)
            positive_vector = self.mlp(positive_vector)
            negative_vector = self.mlp(negative_vector)

        logits1 = vector_pair.pairwise_cosine_similarity(anchor_vector, positive_vector)  # [batch_size, batch_size]

        logits2 = vector_pair.pairwise_cosine_similarity(anchor_vector, negative_vector)  # [batch_size, batch_size]
        # ** 增加 hard negative 的 logits 值
        logits2 = logits2 + torch.eye(batch_size, device=logits2.device) * self.hparams.hard_negative_logits

        logits = torch.cat([logits1, logits2], dim=-1) / self.hparams.temperature  # [batch_size, batch_size * 2]

        target = torch.arange(batch_size).to(logits.device)  # [batch_size, ]
        loss = F.cross_entropy(logits, target)

        self.log("01_train_loss", loss)
        return loss

    def build_val_or_test_dataloader(self, stage: str):

        """ 使用 STS-B 数据集作为测试和验证集 """

        data_dir = os.path.join(utils.DATA_DIR, "sentence_embeddings/STS-B")
        file_name = "cnsd-sts-dev.txt" if stage in ["eval", "val", "dev", "validation"] else "cnsd-sts-test.txt"

        raw_df = pd.read_csv(
            filepath_or_buffer=os.path.join(data_dir, file_name),
            sep=r"\|\|", engine="python",
            names=["source", "sen1", "sen2", "label"], usecols=["label", "sen1", "sen2", ])

        raw_df["sen1_input_ids"] = self.tokenizer(raw_df["sen1"].to_list(), **self.tokenize_kwargs)["input_ids"]
        raw_df["sen2_input_ids"] = self.tokenizer(raw_df["sen2"].to_list(), **self.tokenize_kwargs)["input_ids"]
        raw_df["label"] = raw_df["label"].astype("float32") / 5
        raw_df.drop(columns=["sen1", "sen2"], inplace=True)
        return DataLoader(
            dataset=utils.DataFrameDataset(raw_df),
            collate_fn=utils.DictDataCollator(text_keys=["sen1_input_ids", "sen2_input_ids"], other_keys=["label"]),
            batch_size=self.hparams.eval_or_test_batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

    def build_val_or_test_dataloader_v1(self, stage: str):

        """ 使用 LCQMC 数据集进行测试 """

        file_name = "dev.txt" if stage in ["eval", "val", "dev", "validation"] else "test.txt"
        raw_df = pd.read_csv(
            filepath_or_buffer=os.path.join(utils.DATA_DIR, "sentence_embeddings/LCQMC", file_name),
            sep="\t", header=None, names=["sen1", "sen2", "label"]
        )
        # 理论上说, 验证集和测试集都是经过精细的数据预处理的, 因此这里就不进行数据预处理了
        raw_df["sen1_input_ids"] = self.tokenizer(raw_df["sen1"].to_list(), **self.tokenize_kwargs)["input_ids"]
        raw_df["sen2_input_ids"] = self.tokenizer(raw_df["sen2"].to_list(), **self.tokenize_kwargs)["input_ids"]
        raw_df["label"] = raw_df["label"].astype("float32")
        raw_df.drop(columns=["sen1", "sen2"], inplace=True)
        return DataLoader(
            dataset=utils.DataFrameDataset(raw_df),
            collate_fn=utils.DictDataCollator(text_keys=["sen1_input_ids", "sen2_input_ids"], other_keys=["label"]),
            batch_size=self.hparams.eval_or_test_batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

    def val_dataloader(self): return self.build_val_or_test_dataloader(stage="eval")
    def test_dataloader(self): return self.build_val_or_test_dataloader(stage="test")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        sen1_vector = self.model(batch["sen1_input_ids"])["pooler_output"]
        sen2_vector = self.model(batch["sen2_input_ids"])["pooler_output"]
        scores = vector_pair.paired_cosine_similarity(sen1_vector, sen2_vector)
        return {"pred_scores": scores, "true_scores": batch["label"]}

    test_step = validation_step

    def eval_or_test_epoch_end(self, outputs: List[Dict[str, Tensor]], prefix: str):
        from scipy.stats import pearsonr, spearmanr

        pred_scores = torch.cat([output["pred_scores"] for output in outputs], dim=0).detach().cpu().numpy()
        true_scores = torch.cat([output["true_scores"] for output in outputs], dim=0).detach().cpu().numpy()

        self.log(f"{prefix}_pearsonr", pearsonr(pred_scores, true_scores)[0] * 100)
        self.log(f"{prefix}_spearmanr", spearmanr(pred_scores, true_scores)[0] * 100)

    def validation_epoch_end(self, outputs): return self.eval_or_test_epoch_end(outputs, prefix="02_eval")
    def test_epoch_end(self, outputs): return self.eval_or_test_epoch_end(outputs, prefix="03_test")


if __name__ == '__main__':

    import shutil

    output_dir = os.path.join(utils.ROOT_DIR, "./examples/sentence_embedding/outputs/outputs/s_sim_cse")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    params = HyperParameters()
    system = SupervisedSimCSE(**asdict(params))

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="02_eval_spearmanr", mode="max", dirpath=output_dir, save_top_k=2, every_n_epochs=1)

    trainer = pl.Trainer(
        max_epochs=20, accelerator="cuda", devices=[0, ], default_root_dir=output_dir,
        move_metrics_to_cpu=True, callbacks=[model_checkpoint_callback, ],
    )
    trainer.test(system)
    trainer.fit(system)

    def print_sep():
        print(flush=True)
        print("\n", "======" * 6, flush=True, sep="", end="\n")

    print_sep()
    print("best model path:", model_checkpoint_callback.best_model_path)
    print("best model score:", model_checkpoint_callback.best_model_score.item())
    print_sep()

    trainer.test(ckpt_path="best")
