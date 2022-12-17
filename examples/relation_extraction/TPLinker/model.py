# Author: lqxu

import torch
from torch import Tensor, nn

from core.modules import ConditionalLayerNorm
from core.models import BaseModel, BaseConfig


class TPLinkerConfig(BaseConfig):
    def __init__(self, num_relations: int, use_cln: bool = False, **kwargs):
        super(TPLinkerConfig, self).__init__(**kwargs)
        # 是否使用 CLN 生成 token pair 张量, 如果是否, 则会使用 concat 的方式生成 token pair 张量
        self.use_cln = use_cln
        self.num_relations = num_relations


class HandshakingKernel(nn.Module):

    """ 生成 token pairs 矩阵 """

    def __init__(self, config: TPLinkerConfig):
        super(HandshakingKernel, self).__init__()

        hidden_size = config.bert_config.hidden_size

        self.combine_fn = nn.Linear(hidden_size * 2, hidden_size) if not config.use_cln else None
        self.cln = ConditionalLayerNorm(hidden_size, condition_size=hidden_size) if config.use_cln else None

    def forward(self, token_embeddings: Tensor) -> Tensor:
        num_tokens = token_embeddings.size(1)

        # ## step1: 生成 token pairs 张量
        if self.cln is None:
            # ## method1: 将词向量两两 concat 在一起
            # 注意: torch.cat 方法不接受 broadcasting, 因此需要事先 expand 成相同的 shape
            token_pairs = torch.cat([
                token_embeddings.unsqueeze(2).expand(-1, -1, num_tokens, -1),
                token_embeddings.unsqueeze(1).expand(-1, num_tokens, -1, -1),
            ], dim=-1)  # [batch_size, num_tokens, num_tokens, hidden_size * 2]
        else:
            # ## method2: 将词向量两两之间进行 `反标准化`
            token_pairs = self.cln(torch.unsqueeze(token_embeddings, dim=2), condition=token_embeddings)

        # ## step2: 对 token_pairs 进行 flatten 操作
        bool_mask = torch.ones(size=(num_tokens, num_tokens), dtype=torch.bool, device=token_pairs.device)
        bool_mask.triu_(diagonal=0)  # 保留上三角的数据, 下三角的数据全部变成 0
        # 注意这里的 bool indexing 的用法, 非常重要 !!! 一定要学会哦 !!!
        token_pairs = token_pairs[:, bool_mask]  # [batch_size, num_pairs, hidden_size * 2]
        # num_pairs 的值可以用等差数列求和的方式求取

        # ## step3: 仿射变换 (只有 method1 需要, method2 不需要)
        if self.combine_fn is not None:
            token_pairs = self.combine_fn(token_pairs)
        return token_pairs


class TPLinkerModel(BaseModel):
    def __init__(self, config: TPLinkerConfig):
        super(TPLinkerModel, self).__init__(config)

        self.handshaking = HandshakingKernel(config)

        hidden_size = config.bert_config.hidden_size
        self.entity_classifier = nn.Linear(in_features=hidden_size, out_features=2)
        self.head_classifier = nn.Linear(in_features=hidden_size, out_features=3 * config.num_relations)
        self.tail_classifier = nn.Linear(in_features=hidden_size, out_features=3 * config.num_relations)

    def forward(self, input_ids: Tensor):
        # ## step1: 对词语进行向量化编码
        cal_mask = input_ids.ne(0).float()
        token_vectors = self.bert(input_ids, cal_mask)[0]  # [batch_size, n_tokens, hidden_size]

        # ## step2: 生成 token-pairs 张量
        token_pairs = self.handshaking(token_vectors)  # [batch_size, n_pairs, hidden_size]

        # ## step3: 计算 logits 值
        batch_size, n_pairs, _ = token_pairs.shape
        entity_logits = self.entity_classifier(token_pairs).transpose(1, 2)  # [batch_size, 2, num_pairs]

        # head_logits 和 tail_logits 的 shape 是:   # [batch_size, 3, n_pairs, n_relations]
        head_logits = self.head_classifier(token_pairs).reshape(batch_size, n_pairs, 3, -1).transpose(1, 2)
        tail_logits = self.tail_classifier(token_pairs).reshape(batch_size, n_pairs, 3, -1).transpose(1, 2)

        return entity_logits, head_logits, tail_logits


if __name__ == '__main__':

    batch_size_, num_tokens_, hidden_size_ = 2, 10, 20

    def build_bert_config():
        from transformers import BertConfig

        bert_config_ = BertConfig(hidden_size=hidden_size_)
        return bert_config_

    def test_handshaking_tagger():
        from transformers import BertConfig

        config_ = TPLinkerConfig(num_relations=10, use_cln=True, bert_config=build_bert_config())
        kernel_ = HandshakingKernel(config_)
        input_ = torch.randn(batch_size_, num_tokens_, hidden_size_)
        print(kernel_(input_).shape)

    test_handshaking_tagger()
