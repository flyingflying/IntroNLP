# Author: lqxu

"""
paper: SimCSE: Simple Contrastive Learning of Sentence Embeddings \n
    link: https://aclanthology.org/2021.emnlp-main.552.pdf \n
    code: https://github.com/princeton-nlp/SimCSE \n

无监督 SimCSE, 这里主要借鉴 sentence-transformers 和苏剑林中的实现, 参数参考: https://kexue.fm/archives/8348 \n

如果使用 LCQMC 数据集进行测试, 效果并不好, 测试集上的 spearmanr 会先增后减 (大约在 50-300 步之间达到最大值), 最高能达到 70.10 (验证集最高 64.68)

如果使用 CNSD-STS-B 数据集进行测试, 效果还可以, 测试集上的 spearmanr 不会出现先增后减的情况, 最高能达到 71.82 (验证集最高 76.71)

这里的训练数据来自于数据集的训练集 (没有使用标签), 测试方法和 `Unsupervised STS` 方式是类似的

运行方式: python examples/sentence_embedding/01_u_sim_cse.py

主要的库依赖: pytorch==1.13.0, transformers==4.24.0, pytorch-lightning==1.8.2
"""

import _prepare  # noqa

import os
import random
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
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.3

    # 模型相关
    bert_dropout: float = 0.3
    use_mean_pooling: bool = False
    use_max_pooling: bool = False
    use_first_token_pooling: bool = True
    pooling_with_mlp: bool = False
    temperature: float = 0.05
    add_extra_mlp: bool = False  # 对应原代码里面的 mlp_only_train 参数, 即在训练阶段有 MLP, 在测试阶段没有 MLP

    # 分词器相关
    max_sequence_length: int = 64


class UnsupervisedSimCSE(pl.LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        """ 初始化模型 """

        super(UnsupervisedSimCSE, self).__init__()
        # 保存配置参数
        self.save_hyperparameters(kwargs)

        # 根据参数生成模型
        config = SentenceBertConfig(
            pretrained_name="roberta",
            use_mean_pooling=self.hparams.use_mean_pooling,
            pooling_with_mlp=self.hparams.pooling_with_mlp,
            use_max_pooling=self.hparams.use_max_pooling,
            use_first_token_pooling=self.hparams.use_first_token_pooling,
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

        optimizer_grouped_parameters = self.get_grouped_parameters(
            model=self.model, weight_decay=self.hparams.weight_decay, learning_rate=self.hparams.learning_rate)

        if self.mlp is not None:
            optimizer_grouped_parameters.extend(
                self.get_grouped_parameters(
                    model=self.mlp, weight_decay=self.hparams.weight_decay, learning_rate=self.hparams.learning_rate)
            )

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        # return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_ratio * total_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:

        """ 使用 CNSD-STS-B 数据集作为训练集 """

        data_dir = os.path.join(utils.DATA_DIR, "sentence_embeddings/STS-B/")
        file_name = "cnsd-sts-train.txt"

        df = pd.read_csv(
            os.path.join(data_dir, file_name),
            sep=r"\|\|",  # 注意, 如果 sep 的字符数量超过 1, 那么就是 `正则表达式`
            engine="python",  # 避免警告而指定的
            names=["source", "sen1", "sen2", "label"],
            usecols=["sen1", "sen2"]
        )

        sentences = df["sen1"].to_list() + df["sen2"].to_list()
        random.shuffle(sentences)

        df = pd.DataFrame({
            "sen1_input_ids": self.tokenizer(sentences, **self.tokenize_kwargs)["input_ids"]
        })

        return DataLoader(
            dataset=utils.DataFrameDataset(df),
            collate_fn=utils.DictDataCollator(text_keys=["sen1_input_ids"]),
            batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=8, drop_last=True)

    def train_dataloader_v1(self) -> DataLoader:

        """ 使用 LCQMC 数据集作为训练集 """

        data_dir = os.path.join(utils.DATA_DIR, "sentence_embeddings/LCQMC/")
        file_name = "train.txt"

        df = pd.read_csv(
            filepath_or_buffer=os.path.join(data_dir, file_name),
            sep="\t", header=None, names=["sen1", "sen2", "label"], usecols=["sen1", "sen2"]
        )
        sentences = df["sen1"].to_list() + df["sen2"].to_list()
        sentences = list(set(sentences))  # 去重
        random.shuffle(sentences)

        df = pd.DataFrame({
            "sen1_input_ids": self.tokenizer(sentences, **self.tokenize_kwargs)["input_ids"]
        })

        return DataLoader(
            dataset=utils.DataFrameDataset(df),
            collate_fn=utils.DictDataCollator(text_keys=["sen1_input_ids"]),
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=8,
            # ** 由于使用的是 in-batch negatives 进行训练, 这里最好是直接将无法满足 batch_size 的 mini-batch 直接抛弃掉
            drop_last=True
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        """ 和苏剑林版本的方式是一致的, 采用 `2N-1` 的方式作为负样本 """

        input_ids = batch["sen1_input_ids"]
        batch_size = input_ids.size(0)
        input_ids = torch.cat([input_ids, input_ids], dim=0)  # [batch_size * 2, num_tokens]
        sentence_vector = self.model(input_ids)["pooler_output"]  # [batch_size * 2, hidden_size]

        if self.mlp is not None:
            sentence_vector = self.mlp(sentence_vector)

        # logits: [batch_size * 2, batch_size * 2]
        logits = vector_pair.pairwise_cosine_similarity(sentence_vector, sentence_vector)
        logits = logits / self.hparams.temperature
        # 对角线是其本身, 需要 mask 成无穷小
        logits = logits - torch.eye(logits.size(0), device=logits.device) * 10000.

        # 标签的位置是错开的, 正好隔一个 batch_size 的大小
        target = torch.arange(batch_size, device=logits.device)
        target = torch.cat([target + batch_size, target], dim=0)
        loss = F.cross_entropy(logits, target)

        self.log("01_train_loss", loss)
        return loss

    # noinspection PyUnusedLocal
    def training_step_v1(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        """ 和论文中的方式是一致的, 采用 `N-1` 的方式作为负样本 """

        input_ids = batch["sen1_input_ids"]
        batch_size = input_ids.size(0)
        sentence_vector1 = self.model(input_ids)["pooler_output"]  # [batch_size, hidden_size]
        sentence_vector2 = self.model(input_ids)["pooler_output"]  # [batch_size, hidden_size]

        if self.mlp is not None:
            sentence_vector1 = self.mlp(sentence_vector1)
            sentence_vector2 = self.mlp(sentence_vector2)

        logits = vector_pair.pairwise_cosine_similarity(sentence_vector1, sentence_vector2)  # [batch_size, batch_size]
        logits = logits / self.hparams.temperature  # [batch_size, batch_size]

        # 这里和 阅读理解 的架构差不过, 从一个 mini-batch 中选出正样本
        target = torch.arange(batch_size).to(input_ids.device)
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

        """ 使用 LCQMC 数据集进行验证和测试 """

        file_name = "dev.txt" if stage in ["eval", "val", "dev", "validation"] else "test.txt"
        raw_df = pd.read_csv(
            filepath_or_buffer=os.path.join(utils.DATA_DIR, "sentence_embeddings/LCQMC", file_name),
            sep="\t", header=None, names=["sen1", "sen2", "label"])

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
    """
    tensorboard 运行指令:
        tensorboard --logdir lightning_logs --port 65534 --host 0.0.0.0
    """

    import shutil

    output_dir = os.path.join(utils.ROOT_DIR, "outputs/u_sim_cse")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    params = HyperParameters()
    system = UnsupervisedSimCSE(**asdict(params))

    val_step = 5
    step_validation_callback = pl_callbacks.StepValidationCallback(every_n_steps=val_step)
    progress_bar_callback = pl_callbacks.TQDMCallback()
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="02_eval_spearmanr", mode="max", dirpath=output_dir, save_top_k=2, every_n_train_steps=val_step)
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="02_eval_spearmanr", patience=1000, mode="max", check_on_train_epoch_end=False
    )

    trainer = pl.Trainer(
        max_epochs=5, accelerator="cuda", devices=[1, ], default_root_dir=output_dir, move_metrics_to_cpu=True,
        callbacks=[step_validation_callback, model_checkpoint_callback, progress_bar_callback, early_stopping_callback],
        logger=TensorBoardLogger(save_dir=output_dir), log_every_n_steps=5
        # lightning 显示的步数是 train + validation 的步数, 这里是相隔 train 的步数, 需要注意一下
        # val_check_interval=100,  # 多少步进行 validation
        # 如果设置和 log 相关的参数, 默认的 log 方式会失效, 比方说 tensorboard 会失效, 因此设置时要小心
        # log_every_n_steps=50
    )
    # trainer = pl.Trainer(
    #     max_epochs=1, accelerator="cpu", default_root_dir=output_dir, move_metrics_to_cpu=True,
    #     callbacks=[step_validation_callback, model_checkpoint_callback, progress_bar_callback],
    #     # lightning 显示的步数是 train + validation 的步数, 这里是相隔 train 的步数, 需要注意一下
    #     # val_check_interval=100,  # 多少步进行 validation
    #     # 如果设置和 log 相关的参数, 默认的 log 方式会失效, 比方说 tensorboard 会失效, 因此设置时要小心
    #     # log_every_n_steps=50
    # )
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
