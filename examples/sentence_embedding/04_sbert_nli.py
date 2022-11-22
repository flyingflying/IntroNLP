# Author: lqxu

"""
OCNLI + CNSD-STS-B / LCQMC 训练效果真的不好 !!!

证明了两点：
    1. BERT 之前的 NLP 预训练模型确实不好做;
    2. OCNLI 数据集效果和 SNLI 还是不能比的

运行方式: python examples/sentence_embedding/04_sbert_nli.py

主要的库依赖: pytorch==1.13.0, transformers==4.24.0, pytorch-lightning==1.8.2
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
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from core import utils
from core.utils import vector_pair
from core.trainer import pl_callbacks
from core.models import SentenceBertModel, SentenceBertConfig


@dataclass
class HyperParameters:
    train_batch_size: int = 64
    eval_or_test_batch_size: int = 512
    bert_learning_rate: float = 5e-5
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 模型相关
    use_mean_pooling: bool = True
    use_max_pooling: bool = False
    use_first_token_pooling: bool = False
    pooling_with_mlp: bool = False
    concat_vectors: bool = True  # 通过 concat 融合两个句向量
    sub_abs_vectors: bool = True  # 将两个句向量逐位相减后取绝对值
    mul_vectors: bool = False  # 将两个句向量逐位相乘

    # 分词器相关
    max_sequence_length: int = 64


class SBertNLI(pl.LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        """ 初始化模型 """

        super(SBertNLI, self).__init__()
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
        self.model = SentenceBertModel(config)

        self.label_mapping = {
            "neutral": 0, "entailment": 1, "contradiction": 2, "-": -100
        }

        num_vectors = self.hparams.concat_vectors * 2 + self.hparams.mul_vectors + self.hparams.sub_abs_vectors
        self.mlp = nn.Sequential(
            nn.Linear(self.model.pooler.output_dim * num_vectors, config.bert_config.hidden_size),
            nn.GELU(),
        )
        self.classifier = nn.Linear(config.bert_config.hidden_size, len(self.label_mapping) - 1)

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

        optimizer_grouped_parameters = self.get_grouped_parameters(
            model=self.model, weight_decay=self.hparams.weight_decay, learning_rate=self.hparams.bert_learning_rate)

        optimizer_grouped_parameters.extend(
            self.get_grouped_parameters(
                model=self.mlp, weight_decay=self.hparams.weight_decay, learning_rate=self.hparams.learning_rate)
        )

        optimizer_grouped_parameters.extend(
            self.get_grouped_parameters(
                model=self.classifier, weight_decay=self.hparams.weight_decay,
                learning_rate=self.hparams.learning_rate
            )
        )

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_ratio * total_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        data_dir = os.path.join(utils.DATA_DIR, "sentence_embeddings/OCNLI")
        file_name = "train.50k.jsonl"

        file_path = os.path.join(data_dir, file_name)
        df = pd.read_json(file_path, lines=True)
        df = df.set_index("id")
        df = df[["sentence1", "sentence2", "label"]]

        df["sen1_input_ids"] = self.tokenizer(df["sentence1"].to_list(), **self.tokenize_kwargs)["input_ids"]
        df["sen2_input_ids"] = self.tokenizer(df["sentence2"].to_list(), **self.tokenize_kwargs)["input_ids"]
        df["label"] = df["label"].apply(self.label_mapping.get).astype("int64")
        df.drop(columns=["sentence1", "sentence2", ])

        return DataLoader(
            dataset=utils.DataFrameDataset(df),
            collate_fn=utils.DictDataCollator(
                text_keys=["sen1_input_ids", "sen2_input_ids", ], other_keys=["label"]),
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=8,
            # ** 由于使用的是 in-batch negatives 进行训练, 这里最好是直接将无法满足 batch_size 的 mini-batch 直接抛弃掉
            drop_last=True
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        sen1_vector = self.model(batch["sen1_input_ids"])["pooler_output"]
        sen2_vector = self.model(batch["sen2_input_ids"])["pooler_output"]

        vectors = []
        if self.hparams.concat_vectors:
            vectors.append(sen1_vector)
            vectors.append(sen2_vector)
        if self.hparams.mul_vectors:
            vectors.append(sen1_vector * sen2_vector)
        if self.hparams.sub_abs_vectors:
            vectors.append(torch.abs(sen1_vector - sen2_vector))
        features = torch.concat(vectors, dim=1)  # [batch_size, num_features]
        logits = self.classifier(self.mlp(features))
        loss = F.cross_entropy(logits, batch["label"])
        self.log("01_train_loss", loss)
        return loss

    def build_val_or_test_dataloader(self, stage: str):
        data_dir = os.path.join(utils.DATA_DIR, "sentence_embeddings/STS-B")
        file_name = "cnsd-sts-dev.txt" if stage in ["eval", "val", "dev", "validation"] else "cnsd-sts-test.txt"

        raw_df = pd.read_csv(
            filepath_or_buffer=os.path.join(data_dir, file_name),
            sep=r"\|\|", engine="python",
            names=["source", "sen1", "sen2", "label"], usecols=["label", "sen1", "sen2", ]
        )
        # 理论上说, 验证集和测试集都是经过精细的数据预处理的, 因此这里就不进行数据预处理了
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

    output_dir = os.path.join(utils.ROOT_DIR, "outputs/sbert_nli")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    params = HyperParameters()
    system = SBertNLI(**asdict(params))

    val_step = 5
    step_validation_callback = pl_callbacks.StepValidationCallback(every_n_steps=val_step)
    progress_bar_callback = pl_callbacks.TQDMCallback()
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="02_eval_spearmanr", mode="max", dirpath=output_dir, save_top_k=2, every_n_train_steps=val_step)
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="02_eval_spearmanr", patience=100, mode="max", check_on_train_epoch_end=False
    )

    trainer = pl.Trainer(
        max_epochs=5, accelerator="cuda", devices=[1, ], default_root_dir=output_dir, move_metrics_to_cpu=True,
        callbacks=[step_validation_callback, model_checkpoint_callback, progress_bar_callback, early_stopping_callback],
        logger=TensorBoardLogger(save_dir=output_dir), log_every_n_steps=5
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
