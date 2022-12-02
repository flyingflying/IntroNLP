# Author: lqxu

import _prepare  # noqa

import os
from typing import *
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from core.utils import ROOT_DIR
from core.models import SentenceBertConfig, SentenceBertModel


@dataclass
class HyperParameters:
    # ## 基础设置
    batch_size: int = 20
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    max_epochs: int = 2
    warmup_ratio: float = 0.1

    # ## 预训练模型
    query_encoder_pretrained_name: str = "roberta"
    passage_encoder_pretrained_name: str = "roberta"

    # ## 分词器相关
    # 延用 DuReader Retrieval baseline 中的设置
    max_query_length: int = 32
    max_passage_length: int = 384

    # ## 路径相关
    raw_data_dir: str = os.path.join(ROOT_DIR, "./datasets/du_reader_retrieval/")
    root_dir: str = os.path.join(ROOT_DIR, "./examples/RocketQA/01/")
    use_cached_data: bool = True


class DualEncoder(LightningModule):
    hparams: HyperParameters

    def __init__(self, **kwargs):
        # step1: 初始化参数
        super(DualEncoder, self).__init__()
        self.save_hyperparameters(kwargs)

        # step2: 初始化模型
        self.query_encoder = SentenceBertModel(
            config=SentenceBertConfig(
                pretrained_name=self.hparams.query_encoder_pretrained_name,
                use_mean_pooling=False,
                use_max_pooling=False,
                use_first_token_pooling=True,
                pooling_with_mlp=True
            )
        )
        self.passage_encoder = SentenceBertModel(
            config=SentenceBertConfig(
                pretrained_name=self.hparams.passage_encoder_pretrained_name,
                use_mean_pooling=False, use_max_pooling=False,
                use_first_token_pooling=True, pooling_with_mlp=True
            )
        )

        # step3: 其它设置
        self.hf_data_dir = os.path.join(self.hparams.root_dir, "drr_hf_dual_train_data")
        self.drr_hf_dual_train_dataset = None

    def prepare_data(self) -> None:
        """ 下载和处理数据, 多进程的情况下只会被调用一次 !!! """
        if os.path.exists(self.hf_data_dir) and self.hparams.use_cached_data:
            self.print(f"路径 {self.hf_data_dir} 已经存在, 使用之前处理过的数据 !!!")
            return
        elif os.path.exists(self.hf_data_dir) and not self.hparams.use_cached_data:
            import shutil
            self.print(f"路径 {self.hf_data_dir} 中的所有内容已被移除 !!!")
            shutil.rmtree(self.hf_data_dir)

        from du_reader_retrieval_utils import init_dual_train_data

        init_dual_train_data(
            save_path=self.hf_data_dir, root_data_dir=self.hparams.raw_data_dir, debug=False,
            need_tokenized=True, max_q_len=self.hparams.max_query_length, max_p_len=self.hparams.max_passage_length,
            int_type="int16")

    def setup(self, stage: str) -> None:
        """ 处理和加载数据, 使用 DDP 时这个函数每一个进行都会调用一次 """
        import datasets as hf_datasets
        self.drr_hf_dual_train_dataset = hf_datasets.Dataset.load_from_disk(self.hf_data_dir)

    def train_dataloader(self) -> DataLoader:
        """ 训练的 dataloader """
        from transformers import default_data_collator
        assert self.drr_hf_dual_train_dataset is not None
        return DataLoader(
            dataset=self.drr_hf_dual_train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True, num_workers=8,
            collate_fn=default_data_collator, drop_last=True
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        在这个函数中, q 指 query, p 指 passage, pp 指 positive passage, np 指 negative passage \n
        感谢 PyTorch-Lightning 框架, 让单卡程序和多卡程序的代码可以相互兼容 (直接按照多卡程序写就行, 单卡上能正常执行) \n
        一般情况下, 对于 `数据并行` 来说, 单卡程序和多卡程序的代码是一致的, \n
        但是对于 `模型并行` 来说, 单卡程序和多卡程序的代码还是有一定的差别的, 这里涉及到了简单的 `模型并行` (也就是论文中所说的 cross-batch negatives) \n
        """

        # step1: 对句子进行编码
        q_vectors = self.query_encoder(batch["q_input_ids"])[1]         # [batch_size, hidden_size]
        pp_vectors = self.passage_encoder(batch["pp_input_ids"])[1]     # [batch_size, hidden_size]
        np_vectors = self.passage_encoder(batch["np_input_ids"])[1]     # [batch_size, hidden_size]

        # step2: 收集所有 GPU 上的计算结果
        p_vectors = torch.cat([pp_vectors, np_vectors], dim=0)          # [2 * batch_size, hidden_size]
        p_vectors = self.all_gather(p_vectors, sync_grads=True)         # [world_size, 2 * batch_size, hidden_size]
        batch_size = q_vectors.size(0)
        world_size = p_vectors.size(0)
        # p_vectors = torch.cat(list(p_vectors), dim=0)
        p_vectors = p_vectors.reshape(2 * world_size * batch_size, -1)  # [2 * world_size * batch_size, hidden_size]

        # step3: 计算 logits 值
        logits = torch.mm(q_vectors, p_vectors.T)                       # [batch_size, 2 * world_size * batch_size]

        # step4: 生成标签
        # 目前的数据结构: [第一个 GPU query 的正样本, 第一个 GPU query 的负样本, 第二个 GPU query 的正样本, 第二个 GPU query 的负样本, ...]
        rank = self.global_rank  # 0-based index (按照我目前的理解, self.all_gather 是按照 global_rank 的顺序收集的数据)
        target = torch.arange(
            start=batch_size * (2 * rank), end=batch_size * (2 * rank + 1),
            device=logits.device, dtype=torch.long, requires_grad=False
        )

        # step5: 计算 loss 值
        # 按照我目前的理解, 对于 ddp 策略来说, 每一张卡上的前向传播 (计算 loss) 和 反向传播 (计算梯度) 是独立计算的
        # 然后将不同卡上的梯度同步 (取平均), 最后再分别更新参数
        loss = torch.nn.functional.cross_entropy(logits, target)
        self.log("01_train_loss", loss)
        return loss

    def configure_optimizers(self):
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
        # return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_ratio * total_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]


if __name__ == '__main__':

    import shutil
    from dataclasses import asdict
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    hparams = HyperParameters()

    logger_dir = os.path.join(hparams.root_dir, "lightning_logs")
    if os.path.exists(logger_dir):
        shutil.rmtree(logger_dir)
    logger = TensorBoardLogger(save_dir=hparams.root_dir, name="lightning_logs")

    checkpoint_dir = os.path.join(hparams.root_dir, "checkpoint")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    # 这里只有训练, 没有测试 (测试很耗时间, 这里直接训练, 每一个 epoch 结束保存一下就好啦)
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=1)

    # https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#ddp-optimizations
    ddp_strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)

    trainer = Trainer(
        # 多卡训练, 目前只测试 一机两卡 (nvidia 3090, 24G 显存), 有机会测试多机多卡
        strategy=ddp_strategy, accelerator="gpu", devices=[0, 1],
        # 日志设置 (训练次数较多, 减少 log 记录次数)
        logger=logger, log_every_n_steps=200,
        # 回调函数设置
        callbacks=[checkpoint, ],
        # 其它设置
        max_epochs=hparams.max_epochs, amp_backend="native"
    )

    system = DualEncoder(**asdict(hparams))
    trainer.fit(system)

    """
    hparams = HyperParameters()
    system: DualEncoder = DualEncoder.load_from_checkpoint(
        os.path.join(hparams.root_dir, "./checkpoint/epoch=1-step=44478.ckpt"),
        map_location="cpu", strict=True)

    phase2_root_dir = os.path.join(ROOT_DIR, "./examples/RocketQA/02/")
    system.query_encoder.save_pretrained(
        os.path.join(phase2_root_dir, "model", "query_encoder")
    )
    system.passage_encoder.save_pretrained(
        os.path.join(phase2_root_dir, "model", "passage_encoder")
    )
    """
