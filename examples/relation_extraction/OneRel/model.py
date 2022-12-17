# Author: lqxu

import torch
from torch import Tensor, nn

from core.models import BaseModel, BaseConfig


class OneRelConfig(BaseConfig):
    def __init__(self, num_relations: int, dropout1: float = 0.2, dropout2: float = 0.1, **kwargs):
        super(OneRelConfig, self).__init__(**kwargs)
        # 标签数量
        self.num_relations = num_relations
        # 对词向量进行 dropout
        self.dropout1 = dropout1
        # 对 token pairs 张量进行 dropout
        # 原代码中将其称为 entity pair dropout, 我认为描述不准确, 并没有将某一个 entity pair 给 dropout 掉
        self.dropout2 = dropout2


class OneRelModel(BaseModel):
    def __init__(self, config: OneRelConfig):
        super(OneRelModel, self).__init__(config)

        hidden_size = config.bert_config.hidden_size

        self.dropout1 = nn.Dropout(p=config.dropout1)
        self.dropout2 = nn.Dropout(p=config.dropout2)
        self.dense = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(in_features=hidden_size, out_features=config.num_relations * 4)

    def forward(self, input_ids: Tensor):
        # step1: 对词语进行向量化编码
        cal_mask = input_ids.ne(0).float()
        token_embeddings = self.bert(input_ids=input_ids, attention_mask=cal_mask)[0]
        token_embeddings: Tensor = self.dropout1(token_embeddings)
        batch_size, num_tokens, _ = token_embeddings.shape

        # step2: 生成 token-pairs 张量
        token_pairs = torch.cat([
            token_embeddings.unsqueeze(2).expand(-1, -1, num_tokens, -1),
            token_embeddings.unsqueeze(1).expand(-1, num_tokens, -1, -1),
        ], dim=-1)  # [batch_size, num_tokens, num_tokens, hidden_size * 2]

        # step3: 对 token pairs 进行线性变换, dropout, 并激活
        # 这里很奇怪, 正常情况下这里应该降维, 但是作者却升维
        token_pairs = self.dense(token_pairs)  # [batch_size, num_tokens, num_tokens, hidden_size * 3]
        token_pairs = self.dropout2(token_pairs)
        token_pairs = self.activation(token_pairs)

        # step4: 计算分类的 logits 值
        # logits' shape: [batch_size, 4, num_relations, num_tokens, num_tokens]
        logits = self.classifier(token_pairs).reshape(batch_size, num_tokens, num_tokens, -1, 4).permute(0, 4, 3, 1, 2)

        return logits


if __name__ == '__main__':
    def test_token_pairs_tensor():
        batch_size, num_tokens, hidden_size = 8, 32, 768

        token_embeddings = torch.randn(batch_size, num_tokens, hidden_size)

        token_pairs1 = torch.cat([
            token_embeddings.unsqueeze(2).expand(-1, -1, num_tokens, -1),
            token_embeddings.unsqueeze(1).expand(-1, num_tokens, -1, -1),
        ], dim=-1)

        # OneRel 代码中的写法, 经过测试, 明显比上面慢, 因此就不采用这种方式了
        token_pairs2 = torch.cat([
            token_embeddings.repeat_interleave(num_tokens, dim=1),
            token_embeddings.repeat(1, num_tokens, 1),
        ], dim=-1).reshape(batch_size, num_tokens, num_tokens, -1)

        print(torch.all(token_pairs1 == token_pairs2).item())

    test_token_pairs_tensor()
