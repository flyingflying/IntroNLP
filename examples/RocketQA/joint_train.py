# Author: lqxu

"""
RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking \n

实现 RocketQA V2 的代码, 并没有实际跑, 仅仅做显存测试 \n

paper: https://arxiv.org/pdf/2110.07367.pdf
"""

import _prepare  # noqa

from typing import *
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from pytorch_lightning import LightningModule

from core.models import SentenceBertModel, CLSBertModel
from core.models import SentenceBertConfig, CLSBertConfig


@dataclass
class HyperParameters:
    list_size: int = 24


class RocketQAV2(LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(RocketQAV2, self).__init__()

        self.save_hyperparameters(kwargs)

        self.query_encoder = SentenceBertModel(
            SentenceBertConfig(
                use_mean_pooling=False, use_max_pooling=False, use_first_token_pooling=True, pooling_with_mlp=True))
        self.passage_encoder = SentenceBertModel(
            SentenceBertConfig(
                use_mean_pooling=False, use_max_pooling=False, use_first_token_pooling=True, pooling_with_mlp=True))
        self.cross_encoder = CLSBertModel(CLSBertConfig(num_classes=1))

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        # shape: [batch_size * list_size, hidden_size]
        q_input_ids, p_input_ids, c_input_ids = batch["q_input_ids"], batch["p_input_ids"], batch["c_input_ids"]
        list_size = self.hparams.list_size
        batch_size = c_input_ids.size(0) // list_size

        """
        dual encoder 编码后的句向量需要两两点乘得到 logits 值 \n
        注意: 按照 RocketQA 数据处理的方式, 对于 q_input_ids 来说, 其 shape 是 [batch_size * list_size, q_seq_len], 在一个 list 中的
        所有 query 句子是相同的。由于 bert 模型有 dropout, 因此相同的句子输入会产生不同的句向量结果, 这增加了模型训练的难度, 因此这里我也采用了相同
        的方式 (原本打算一个 query 只编码一次, 现在要编码 list_size 次, 对显卡的显存要求更高了) \n
        """
        q_vectors = self.query_encoder(q_input_ids)[1].reshape(batch_size, list_size, -1)
        p_vectors = self.passage_encoder(p_input_ids)[1].reshape(batch_size, list_size, -1)
        de_logits = torch.sum(q_vectors * p_vectors, dim=-1)  # [batch_size, list_size]

        """ cross encoder 出来的直接是 logits 值 """
        ce_logits = self.cross_encoder(c_input_ids).reshape(batch_size, list_size)  # [batch_size, list_size]

        """ supervision 损失函数: 每一个 list 中第一个句子是正样本, 其它句子是负样本, 这里使用的是对比学习的 loss 值计算方式 """
        target = torch.zeros(batch_size, dtype=torch.long, device=ce_logits.device)
        supervision_loss = nn.functional.cross_entropy(ce_logits, target)  # 里面包含了对一个 list 所有句子的归一化

        """
        KL-divergence 损失函数: 
        
        在 PyTorch 中, KLDivLoss 中, input 是模型预测的分布, target 是样本实际的分布, 也就是 KL(target || input) \n
        在论文中, KL 散度损失函数的定义是 KL(de_probs || ce_probs), 因此 input 是 ce_probs, target 是 de_probs \n
        但是在 RocketQA 代码中, 其实现的是: input 是 de_probs, target 是 ce_probs, 我认为这样更合理, 即 ce 指导 de 的训练, 这里采用这种方式 \n
        
        在 PyTorch 中, KLDivLoss 的输入不是 logits 值, 而是 probs 值, 并且希望 probs 值在 log 空间中, 那么就需要将 logits 值用 log_softmax 先转换一下 \n
        target 值如果在 log 空间中, 则参数 log_target 设置成 True, 如果不再 log 空间中, 则参数 log_target 设置成 False \n
        """
        de_log_probs = nn.functional.log_softmax(de_logits, dim=-1)
        ce_probs = nn.functional.softmax(ce_logits, dim=-1)
        kl_loss = nn.functional.kl_div(input=de_log_probs, target=ce_probs, reduction="batchmean", log_target=False)

        loss = supervision_loss + kl_loss
        # self.log("01_train_loss", loss)
        return loss


if __name__ == '__main__':

    """ 对于 24G 的显卡, batch size 为 1 勉强能跑起来 """

    from dataclasses import asdict

    hparams = HyperParameters()
    system = RocketQAV2(**asdict(hparams))

    device = "cuda:1"
    batch_size_ = 1
    q_len, p_len = 32, 384
    tensor_size = batch_size_ * hparams.list_size
    c_len = q_len + p_len

    system.to(device).train()

    batch_ = {
        "q_input_ids": torch.randint(low=0, high=20000, size=(tensor_size, q_len)).to(device),
        "p_input_ids": torch.randint(low=0, high=20000, size=(tensor_size, p_len)).to(device),
        "c_input_ids": torch.randint(low=0, high=20000, size=(tensor_size, c_len)).to(device)
    }

    loss_ = system.training_step(batch_, 0)
    loss_.backward()
    print(loss_)
