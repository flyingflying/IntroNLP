# Author: lqxu

from typing import *

from torch import nn, Tensor
from .base import BaseConfig, BaseModel

__all__ = ["CLSBertConfig", "CLSBertModel"]


class CLSBertConfig(BaseConfig):
    def __init__(self, num_classes: int, dropout: float = 0.1, **kwargs):
        super(CLSBertConfig, self).__init__(**kwargs)

        self.num_classes = num_classes  # 类别数
        self.dropout = dropout


class CLSBertModel(BaseModel):
    def __init__(self, config: CLSBertConfig, bert_kwargs: Dict = None):
        super(CLSBertModel, self).__init__(config, bert_kwargs)

        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(config.bert_config.hidden_size, config.bert_config.hidden_size),
            nn.Tanh(),
        )

        self.classifier = nn.Linear(in_features=config.bert_config.hidden_size, out_features=config.num_classes)

    def forward(self, input_ids: Tensor, attention_masks: Tensor = None, token_type_ids: Tensor = None):
        if attention_masks is None:
            attention_masks = input_ids.ne(0)
        attention_masks = attention_masks.float()
        # [batch_size, num_tokens, hidden_size]
        word_embeds = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)[0]
        sen_embeds = word_embeds[:, 0, :]  # [batch_size, hidden_size]
        sen_embeds = self.mlp(sen_embeds)
        # [batch_size, num_classes]
        logits = self.classifier(sen_embeds)
        return logits
