# Author: lqxu

from typing import *

import torch
from torch import Tensor, nn

from core.models import BaseModel, BaseConfig


class PRGCConfig(BaseConfig):
    def __init__(self, num_relations: int, classifier_dropout: float = 0.3, relation_threshold: float = 0.5, **kwargs):
        super(PRGCConfig, self).__init__(**kwargs)

        self.num_relations = num_relations  # 预测关系的数量
        self.classifier_dropout = classifier_dropout
        self.relation_threshold = relation_threshold


class PRGCModel(BaseModel):
    def __init__(self, config: PRGCConfig):
        super(PRGCModel, self).__init__(config)
        hidden_size = config.bert_config.hidden_size

        self.relation_threshold = config.relation_threshold

        self.relation_judgement = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size // 2, config.num_relations)
        )

        self.relation_embeddings = nn.Embedding(num_embeddings=config.num_relations, embedding_dim=hidden_size)

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Dropout(config.classifier_dropout)
        )

        # 采用的是 BIO 的标注方式
        self.subject_classifier = nn.Linear(hidden_size // 2, 3)
        self.object_classifier = nn.Linear(hidden_size // 2, 3)

        self.global_correspondence = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(), nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, 1)
        )

    @staticmethod
    def get_entity(sequence: List[int]) -> List[Tuple[int, int]]:
        """ 常规的 BIO 解码, 只是这里定死了 0 表示 `O`, 1 表示 `B`, 2 表示 `I` """
        entities: List[Tuple[int, int]] = []
        entity_start_index: int = -1

        for index, token_tag in enumerate(sequence):
            if token_tag == 0:
                if entity_start_index != -1:
                    entities.append((entity_start_index, index - 1))
                    entity_start_index = -1
                continue

            if token_tag == 1:
                if entity_start_index != -1:
                    entities.append((entity_start_index, index - 1))
                entity_start_index = index
                continue

            if entity_start_index == -1:
                entity_start_index = index

        if entity_start_index != -1:
            entities.append((entity_start_index, len(sequence) - 1))

        return entities

    @torch.no_grad()
    def decode(self, input_ids: Tensor):
        # ## step1: 词语向量化
        bool_mask = input_ids.eq(0)  # [batch_size, num_tokens]
        cal_mask = input_ids.ne(0).float()  # [batch_size, num_tokens]
        token_vectors = self.bert(input_ids, cal_mask)[0]  # [batch_size, num_tokens, hidden_size]
        batch_size, num_tokens, _ = token_vectors.shape

        # ## step2: potential relation prediction
        # 2.1 平均池化获得句向量
        seq_cal_mask = cal_mask.unsqueeze(-1)  # [batch_size, num_tokens, 1]
        sentence_vector = torch.sum(token_vectors * seq_cal_mask, dim=1) / torch.sum(seq_cal_mask, dim=1)
        # 2.2 进行 `关系` 预测
        relation_logits = self.relation_judgement(sentence_vector)  # [batch_size, num_relations]
        # 2.3 生成下一步预测的张量
        batch_indices, relation_indices = torch.where(torch.sigmoid(relation_logits) > self.relation_threshold)
        new_token_vectors = token_vectors[batch_indices]  # [num_results, num_tokens, hidden_size]
        relation_vectors = self.relation_embeddings(relation_indices)  # [num_results, hidden_size]

        # ## step3: relation-specific sequence tagging
        # 原版这里还有 concat 的方式, 这里就先省略了
        new_token_vectors = new_token_vectors + relation_vectors.unsqueeze(1)  # [num_results, num_tokens, hidden_size]
        new_token_vectors = self.dense(new_token_vectors)  # [num_results, num_tokens, hidden_size // 2]

        subject_logits = self.subject_classifier(new_token_vectors)  # [num_results, num_tokens, 3]
        object_logits = self.object_classifier(new_token_vectors)    # [num_results, num_tokens, 3]

        subject_predictions = torch.argmax(subject_logits, dim=-1)   # [num_results, num_tokens]
        object_predictions = torch.argmax(object_logits, dim=-1)     # [num_results, num_tokens]

        new_bool_mask = bool_mask[batch_indices]  # [num_results, num_tokens]
        subject_predictions[new_bool_mask] = 0
        object_predictions[new_bool_mask] = 0

        # ## step4: global correspondence
        correspondence_logits = self.global_correspondence(
            torch.cat([
                token_vectors.unsqueeze(2).expand(-1, -1, num_tokens, -1),
                token_vectors.unsqueeze(1).expand(-1, num_tokens, -1, -1)
            ], dim=-1)
        ).squeeze(-1)  # [batch_size, num_tokens, num_tokens]

        # 上面的 subject_predictions 和 object_predictions 已经 mask 过了, 因此这里就不需要再 mask 了
        correspondence = correspondence_logits > 0

        # ## step5: 解码
        num_results = batch_indices.size(0)
        ret = [set() for _ in range(batch_size)]

        for idx in range(num_results):
            batch_idx = batch_indices[idx].item()
            rl = relation_indices[idx].item()
            subjects = self.get_entity(subject_predictions[idx].tolist())
            objects = self.get_entity(object_predictions[idx].tolist())

            for sh, st in subjects:
                for oh, ot in objects:
                    if correspondence[batch_idx, sh, oh].item():
                        ret[batch_idx].add((sh, st, rl, oh, ot))  # noqa

        return ret


if __name__ == '__main__':
    from transformers import set_seed

    set_seed(45)

    config_ = PRGCConfig(2, 0.3)
    model_ = PRGCModel(config_)
    print(model_.decode(torch.tensor([[1, 2, 3, 4, 5]])))
