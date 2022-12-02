# Author: lqxu

"""
PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval

paper: https://aclanthology.org/2021.findings-acl.191.pdf
"""

from typing import *
from dataclasses import dataclass

import torch
from torch import Tensor
from pytorch_lightning import LightningModule

from core.models import SentenceBertModel, SentenceBertConfig


@dataclass
class HyperParameters:
    # loss 计算相关的参数
    is_pretrained: bool = True  # pretrained 表示 query-centric 和 passage-centric 一起训练, 非 pretrained 只训练 query-centric
    alpha: float = 0.1  # passage-centric loss 占最终 loss 的比例, 按照论文所说, 0.1 的效果最好


class PAIR(LightningModule):

    hparams: HyperParameters

    def __init__(self, **kwargs):
        super(PAIR, self).__init__()
        self.save_hyperparameters(kwargs)

        self.query_passage_encoder = SentenceBertModel(
            SentenceBertConfig(
                use_mean_pooling=False, use_max_pooling=False,
                use_first_token_pooling=True, pooling_with_mlp=True
            )
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        q_input_ids, pp_input_ids, np_input_ids = batch["q_input_ids"], batch["pp_input_ids"], batch["np_input_ids"]
        batch_size = q_input_ids.size(0)

        q_embeds = self.query_passage_encoder(q_input_ids)[1]  # [batch_size, hidden_size]
        pp_embeds = self.query_passage_encoder(pp_input_ids)[1]  # [batch_size, hidden_size]
        np_embeds = self.query_passage_encoder(np_input_ids)[1]  # [batch_size, hidden_size]

        """ 计算 query-centric loss """
        # 如果要实现 cross-batch negatives, 则要对 p_embeds 进行 all_gather 操作
        p_embeds = torch.cat([pp_embeds, np_embeds], dim=0)  # [batch_size * 2, hidden_size]
        logits = torch.mm(q_embeds, p_embeds.T)  # [batch_size, batch_size * 2]
        target = torch.arange(batch_size, device=logits.device, dtype=torch.long)  # [batch_size, ]
        loss = torch.nn.functional.cross_entropy(logits, target)

        """ 计算 passage-centric loss """
        if self.hparams.is_pretrained:
            alpha = self.hparams.alpha
            # 如果要实现 cross-batch negatives, 则要对 q_np_embeds 进行 all_gather 操作
            q_np_embeds = torch.cat([q_embeds, np_embeds], dim=0)  # [batch_size * 2, hidden_size]
            logits = torch.mm(pp_embeds, q_np_embeds.T)  # [batch_size, batch_size * 2]
            pc_loss = torch.nn.functional.cross_entropy(logits, target)
            loss = (1 - alpha) * loss + alpha * pc_loss

        return loss
