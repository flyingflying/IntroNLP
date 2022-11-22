# Author: lqxu

"""
paper: ESimCSE: Enhanced Sample Building Method for Contrastive Learning of Unsupervised Sentence Embedding \n
    link: https://arxiv.org/pdf/2109.04380.pdf \n
    code: https://github.com/caskcsg/sentemb/tree/main/ESimCSE \n

这里的实现主要参考了何凯明大神的代码: https://github.com/facebookresearch/moco \n

在 LCQMC 上的效果不好, 训练结果会先增后减,
CNSD-STS-B 上的效果: test_spearmanr=74.20; eval_spearmanr=78.86

运行方式: python examples/sentence_embedding/03_e_sim_cse.py

主要的依赖库: pytorch==1.13.0, transformers==4.24.0, pytorch-lightning==1.8.2
"""

import _prepare  # noqa

import os
import random
from typing import *
from dataclasses import dataclass, asdict

import torch
import pandas as pd
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
    warmup_ratio: float = 0.1

    # 模型相关
    bert_dropout: float = 0.15
    use_mean_pooling: bool = False
    use_max_pooling: bool = False
    use_first_token_pooling: bool = True
    pooling_with_mlp: bool = False
    temperature: float = 0.05

    # 数据增强相关, 这里使用的是 word repetition 增强方式, 原代码还使用了添加 stop words 作为数据增强的方式
    repetition_rate: float = 0.3
    # 动量编码器相关的参数
    momentum: float = 0.995
    queue_size: int = 640  # 负样本的个数 (对应原代码的 neg_size, 按照论文中的说法, 这个值应该是 batch_size 的 2.5 倍)

    # 分词器相关
    max_sequence_length: int = 64


class ESimCSE(pl.LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        """ 初始化模型 """

        super(ESimCSE, self).__init__()
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

        # 动量编码器 (让其 dropout 值为 0.0)
        momentum_bert_kwargs = {
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
        }
        self.momentum_model = SentenceBertModel(config, momentum_bert_kwargs)
        # 貌似不用复制也是可以的 (因为加载的是预训练模型), 保险起见, 还是复制了
        for param, m_param in zip(self.model.parameters(), self.momentum_model.parameters()):
            m_param.data.copy_(param.data)
            # 其实不用设置成 False 也是可以的, 只要别把其放入 optimizer 中即可
            m_param.requires_grad = False

        # 初始化队列
        self.queue: Tensor  # [queue_size, hidden_size]
        self.queue_ptr: Tensor  # [1, ]
        self.register_buffer("queue", torch.randn(self.hparams.queue_size, config.bert_config.hidden_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # 为了保存到模型中

        # 分词器
        self.tokenizer = utils.get_default_tokenizer()
        self.tokenize_kwargs = {
            "max_length": self.hparams.max_sequence_length,
            "padding": "max_length", "truncation": True,
            "return_attention_mask": False, "return_token_type_ids": False
        }

    def configure_optimizers(self):
        """
        采用 transformers 默认的 optimizer 和 lr_scheduler \n
        reference:
            https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
        """

        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
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

        def create_positive_sentence(sen: str):
            length = random.randint(1, len(sen))
            rep_num = int(length * self.hparams.repetition_rate)
            rep_num = max(1, rep_num)  # 设置最小为 1

            sen_list = list(sen)
            for idx in random.sample(range(len(sen_list)), rep_num):
                sen_list[idx] = sen_list[idx] * 2  # noqa

            return "".join(sen_list)

        data_dir = os.path.join(utils.DATA_DIR, "sentence_embeddings/STS-B")
        file_name = "cnsd-sts-train.txt"

        df = pd.read_csv(
            os.path.join(data_dir, file_name),
            sep=r"\|\|",  # 注意, 如果 sep 的字符数量超过 1, 那么就是 `正则表达式`
            engine="python",  # 避免警告而指定的
            names=["source", "sen1", "sen2", "label"],
            usecols=["sen1", "sen2"]
        )
        queries = df["sen1"].to_list() + df["sen2"].to_list()
        keys = [create_positive_sentence(query) for query in queries]

        df = pd.DataFrame({
            "query_input_ids": self.tokenizer(queries, **self.tokenize_kwargs)["input_ids"],
            "key_input_ids": self.tokenizer(keys, **self.tokenize_kwargs)["input_ids"],
        })

        return DataLoader(
            dataset=utils.DataFrameDataset(df),
            collate_fn=utils.DictDataCollator(text_keys=["query_input_ids", "key_input_ids"]),
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        """
        参考何凯明大神的 MoCo 代码: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
        """

        # step1: 对 query 进行编码
        query_input_ids = batch["query_input_ids"]
        query_vector = self.model(query_input_ids)["pooler_output"]  # [batch_size, hidden_size]

        # step2: 更新动量编码器, 并对 key 进行编码
        # 不用担心初始队列的问题, 随机向量作为负样本是 ok 的
        with torch.no_grad():
            momentum = self.hparams.momentum
            for param, momentum_param in zip(self.model.parameters(), self.momentum_model.parameters()):
                momentum_param.data = momentum_param.data * momentum + param.data * (1 - momentum)

            key_input_ids = batch["key_input_ids"]
            key_vector = self.momentum_model(key_input_ids)["pooler_output"]

        # step3: 计算 logits 值
        # positive_logits: [batch_size, 1]
        positive_logits = vector_pair.paired_cosine_similarity(query_vector, key_vector, keepdim=True)
        # negative_logits: [batch_size, queue_size]
        negative_logits = vector_pair.pairwise_cosine_similarity(query_vector, self.queue.detach().clone())
        # logits: [batch_size, queue_size+1]
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits = logits / self.hparams.temperature

        # step4: 计算 loss 值
        batch_size = logits.size(0)
        target = torch.zeros(batch_size, dtype=torch.long).to(logits.device)
        loss = F.cross_entropy(logits, target)

        # step5: 更新队列
        start_idx = self.queue_ptr.item()
        end_idx = start_idx + batch_size
        if end_idx <= self.hparams.queue_size:
            self.queue[start_idx:end_idx, :] = key_vector
        else:
            extra_size = end_idx - self.hparams.queue_size
            self.queue[start_idx:, :] = key_vector[:batch_size-extra_size]
            self.queue[:extra_size, :] = key_vector[batch_size-extra_size:]
        self.queue_ptr[0] = end_idx % self.hparams.queue_size

        # step6: 返回 loss
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

    output_dir = os.path.join(utils.ROOT_DIR, "outputs/e_sim_cse")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    params = HyperParameters()
    system = ESimCSE(**asdict(params))

    val_step = 5
    step_validation_callback = pl_callbacks.StepValidationCallback(every_n_steps=val_step)
    progress_bar_callback = pl_callbacks.TQDMCallback()
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="02_eval_spearmanr", mode="max", dirpath=output_dir, save_top_k=2, every_n_train_steps=val_step)
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="02_eval_spearmanr", patience=200, mode="max", check_on_train_epoch_end=False
    )

    trainer = pl.Trainer(
        max_epochs=3, accelerator="cuda", devices=[0, ], default_root_dir=output_dir, move_metrics_to_cpu=True,
        callbacks=[step_validation_callback, model_checkpoint_callback, progress_bar_callback, early_stopping_callback],
        log_every_n_steps=5, logger=TensorBoardLogger(save_dir=output_dir)
    )
    # trainer.test(system)
    trainer.fit(system)

    def print_sep():
        print(flush=True)
        print("\n", "======" * 6, flush=True, sep="", end="\n")

    print_sep()
    print("best model path:", model_checkpoint_callback.best_model_path)
    print("best model score:", model_checkpoint_callback.best_model_score.item())
    print_sep()

    trainer.test(ckpt_path="best")
