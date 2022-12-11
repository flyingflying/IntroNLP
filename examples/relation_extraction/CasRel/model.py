# Author: lqxu

from typing import *

import torch
from torch import Tensor, nn

from core.models import BaseModel, BaseConfig

__all__ = ["CasRelConfig", "CasRelModel"]


class CasRelConfig(BaseConfig):
    def __init__(self, relation_labels: List[str], **kwargs):
        self.relation_labels = relation_labels
        super(CasRelConfig, self).__init__(**kwargs)


class CasRelModel(BaseModel):
    def __init__(self, config: CasRelConfig):
        super(CasRelModel, self).__init__(config)

        hidden_size = config.bert_config.hidden_size
        self.num_labels = len(config.relation_labels)
        self.subj_head_tagger = nn.Linear(hidden_size, 1)
        self.subj_tail_tagger = nn.Linear(hidden_size, 1)
        self.rel_obj_head_tagger = nn.Linear(hidden_size, self.num_labels)
        self.rel_obj_tail_tagger = nn.Linear(hidden_size, self.num_labels)

    @staticmethod
    def span_decode(head_logits: Tensor, tail_logits: Tensor) -> Tuple[List[int], List[int]]:
        """
        解码基于 span 的 NER:
            1. 根据 head_logits 找到所有的 head index
            2. 遍历每一个 head index, 根据 tail_logits, 找到离其最近的右边的 tail index, 构成一个实体, 如果没找到, 则认为不构成实体
        这里认为传入的 head_logits 和 tail_logits 都是一维向量, 即 `[num_tokens, ]`
        """

        # 对于 torch.where 返回的第一个 Tensor, 其一定是按照从小到大排序的
        candidate_head_indices = torch.where(head_logits > 0)[0].tolist()
        candidate_tail_indices = torch.where(tail_logits > 0)[0].tolist()

        head_indices, tail_indices = [], []
        for head_index in candidate_head_indices:
            for tail_index in candidate_tail_indices:
                if tail_index < head_index:
                    continue
                head_indices.append(head_index)
                tail_indices.append(tail_index)
                break

        return head_indices, tail_indices

    def forward(self, input_ids: Tensor) -> List[Tuple[int, int, int, int, int]]:

        """
        只是解码过程, 不包含训练过程, 返回的是 (subj_head_idx, subj_tail_idx, rel_idx, obj_head_idx, obj_tail_idx) \n

        注意这里要求 batch_size 必须为 1, 同时 input_ids 中没有 sequence padding, 也就是 0 \n

        这个模型虽然是联合模型, 但是解码的方式和管道模型差不多, 没有办法并行化计算 (主要原因是每一个句子的 subject 数不是固定的, 导致很难向量化计算)。
        因此, 我用更 Python 化的方式来写代码了, 想要高效, 解码必须用 C++ 写。当然, 目前的实现是不支持用 `torch.jit.trace` 导出的。
        """

        # input_ids: [batch_size, num_tokens]
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("CasRel 模型解码不能并行化计算 ... ")
        sro_list = []

        # step1: 用 BERT 模型编码词向量
        token_embeddings = self.bert(input_ids)[0][0]  # [num_tokens, hidden_size]

        # step2: 预测 subject
        subj_heads, subj_tails = self.span_decode(
            self.subj_head_tagger(token_embeddings).squeeze(),  # [num_tokens,]
            self.subj_tail_tagger(token_embeddings).squeeze()   # [num_tokens,]
        )
        if len(subj_heads) == 0:  # 没有 subject, 直接返回
            return sro_list

        # step3: 预测 relation_label 和 object
        # 3.1 将 subject 的相关信息融入 token_embeddings 中, 采用加法的形式
        subj_head_embeddings = token_embeddings[subj_heads]  # [num_subjects, hidden_size]
        subj_tail_embeddings = token_embeddings[subj_tails]  # [num_subjects, hidden_size]
        subj_embeddings = (subj_head_embeddings + subj_tail_embeddings) / 2  # [num_subjects, hidden_size]
        token_embeddings = token_embeddings + subj_embeddings.unsqueeze(1)  # [num_subjects, num_tokens, hidden_size]

        # 3.2 预测
        rel_obj_head_logits = self.rel_obj_head_tagger(token_embeddings)  # [num_subjects, num_tokens, num_labels]
        rel_obj_tail_logits = self.rel_obj_tail_tagger(token_embeddings)  # [num_subjects, num_tokens, num_labels]
        for subject_idx, (subj_head, subj_tail) in enumerate(zip(subj_heads, subj_tails)):
            for rel_idx in range(self.num_labels):
                obj_heads, obj_tails = self.span_decode(
                    rel_obj_head_logits[subject_idx, :, rel_idx],
                    rel_obj_tail_logits[subject_idx, :, rel_idx]
                )
                sro_list.extend([
                    (subj_head, subj_tail, rel_idx, obj_head, obj_tail)
                    for obj_head, obj_tail in zip(obj_heads, obj_tails)
                ])

        return sro_list


if __name__ == '__main__':

    from transformers import set_seed

    from core.utils import get_default_tokenizer

    config_ = CasRelConfig(
        relation_labels=["父亲", "母亲"]
    )

    set_seed(42)
    model = CasRelModel(config_).eval().to("cpu")

    tokenizer = get_default_tokenizer()

    input_ids_ = tokenizer("小王的父亲是小明", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        print(model(input_ids_))
