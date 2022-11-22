# Author: lqxu

""" sentence bert 的封装, 几乎所有的基于 bert 的句向量架构都是这样的 """

from typing import *

import torch
from torch import Tensor, nn

from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPastAndCrossAttentions
)

from .base import BaseConfig, BaseModel

__all__ = ["SentenceBertConfig", "SentenceBertModel"]


class SentenceBertConfig(BaseConfig):
    def __init__(
            self, use_mean_pooling: bool = True, use_max_pooling: bool = False,
            use_first_token_pooling: bool = False, pooling_with_mlp: bool = False, **kwargs):

        self.use_max_pooling = use_max_pooling
        self.use_mean_pooling = use_mean_pooling
        self.use_first_token_pooling = use_first_token_pooling
        self.pooling_with_mlp = pooling_with_mlp

        super(SentenceBertConfig, self).__init__(**kwargs)


class PoolingLayer(nn.Module):
    def __init__(self, config: SentenceBertConfig):
        num_vectors = config.use_first_token_pooling + config.use_max_pooling + config.use_mean_pooling
        if num_vectors == 0:
            raise ValueError("池化层至少要有一种方式")
        self.output_dim = num_vectors * config.bert_config.hidden_size
        self.use_first_token_pooling = config.use_first_token_pooling
        self.use_max_pooling = config.use_max_pooling
        self.use_mean_pooling = config.use_mean_pooling

        super(PoolingLayer, self).__init__()
        if config.pooling_with_mlp:
            self.perceptron = nn.Sequential(nn.Linear(self.output_dim, config.bert_config.hidden_size), nn.Tanh())
        else:
            self.perceptron = None

    def forward(self, token_embeddings: Tensor, mask: Tensor) -> Tensor:
        """
        :param token_embeddings: [batch_size, seq_len, input_size]
        :param mask: [batch_size, seq_len, ] 其中 0 表示 mask 掉 (不参与后续计算), 1 表示不 mask 掉 (参与后续计算)
        :return: 池化后的张量, [batch_size, self.output_dim]
        """
        # token_embeddings = torch.stack(
        #     [hidden_states[layer_index] for layer_index in self.layers_avg_index], dim=0).mean(dim=0)
        output_vectors = []
        if self.use_first_token_pooling:
            output_vectors.append(token_embeddings[:, 0, :])
        if self.use_max_pooling:
            _token_embeddings = token_embeddings + (1 - torch.unsqueeze(mask, dim=-1)) * (-10000.)
            output_vectors.append(torch.max(_token_embeddings, dim=1)[0])
        if self.use_mean_pooling:
            _token_embeddings = token_embeddings * torch.unsqueeze(mask, dim=-1)
            output_vectors.append(torch.sum(_token_embeddings, dim=1) / torch.sum(mask, dim=1, keepdim=True))
        ret = torch.cat(output_vectors, dim=1)

        if self.perceptron is not None:
            ret = self.perceptron(ret)
        return ret


class SentenceBertModel(BaseModel):
    config_class = SentenceBertConfig

    def __init__(self, config: SentenceBertConfig, bert_kwargs: Dict = None):
        super(SentenceBertModel, self).__init__(config, bert_kwargs)
        self.config = config
        self.pooler = PoolingLayer(config)

    def forward(
            self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            output_attentions=None, output_hidden_states=None, return_dict=None):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Union[Tensor, BaseModelOutputWithPooling]

        # step1: 从 config 文件中修正参数值
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.return_dict

        # step2: 准备 attention_mask 和 head_mask 参数
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        attention_mask = attention_mask.float()
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        head_mask = self.get_head_mask(head_mask, self.config.bert_config.num_hidden_layers)

        # step3: 跑模型
        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.bert.encoder(
            embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output, attention_mask)

        # step4: 返回结果
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )
