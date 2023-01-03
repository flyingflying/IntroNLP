# Author: lqxu

""" GPLinker 模型模块 """

from torch import nn, Tensor

from core.models import BaseModel, BaseConfig
from core.modules import EfficientGlobalPointer


class GPLinkerConfig(BaseConfig):
    def __init__(self, num_relations: int, **kwargs):
        super(GPLinkerConfig, self).__init__(**kwargs)

        self.num_relations = num_relations


class GPLinkerModel(BaseModel):
    """ Reference: https://www.kexue.fm/archives/8888 """

    def __init__(self, config: GPLinkerConfig):
        super(GPLinkerModel, self).__init__(config)

        # 实体识别需要 rope
        self.entity_classifier = EfficientGlobalPointer(
            config.bert_config, num_tags=2, head_size=64, use_rope=True)

        # 链接识别不需要 rope
        self.head_classifier = EfficientGlobalPointer(
            config.bert_config, num_tags=config.num_relations, head_size=64, use_rope=False)
        self.tail_classifier = EfficientGlobalPointer(
            config.bert_config, num_tags=config.num_relations, head_size=64, use_rope=False)

    def forward(self, input_ids: Tensor):
        # step1: 对词语进行向量化编码
        cal_mask = input_ids.ne(0).float()
        token_vectors = self.bert(input_ids, attention_mask=cal_mask)[0]  # [batch_size, n_tokens, hidden_size]

        # step2: 预测实体
        entity_logits = self.entity_classifier(token_vectors)  # [batch_size, 2, n_tokens, n_tokens]
        head_logits = self.head_classifier(token_vectors)  # [batch_size, n_relations, n_tokens, n_tokens]
        tail_logits = self.tail_classifier(token_vectors)  # [batch_size, n_relations, n_tokens, n_tokens]

        return entity_logits, head_logits, tail_logits
