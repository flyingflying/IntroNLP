# Author: lqxu

""" 将三元组转化为 TPLinker 所需要的形式 """

from typing import *
from collections import defaultdict

import torch
from torch import Tensor

# 三元组类型: (subj_head, subj_tail, rel_label, obj_head, obj_tail)
_SRO_TYPING = Tuple[int, int, int, int, int]

__all__ = ["TPLinkerScheme", ]


class TPLinkerScheme:
    def __init__(self, num_relations: int, max_num_tokens: int = 512):
        self._num_relations = num_relations
        self._max_num_tokens = max_num_tokens

        # m 指的是 matrix, 即以二元组的方式来描述位置, 比方说: (head_index, tail_index)
        # s 指的是 shaking sequence, 即将矩阵的上三角 flatten 成一个 sequence
        self._s_to_m = [(i, j) for i in range(max_num_tokens) for j in range(i, max_num_tokens)]
        self._m_to_s = {m: s for s, m in enumerate(self._s_to_m)}

        self._num_pairs = len(self._s_to_m)

    def encode(self, sro_set: Set[_SRO_TYPING]) -> Tuple[Tensor, Tensor, Tensor]:
        """ 将 `三元组` 转化为模型训练所需要的数据 """
        # entity head to entity tail
        entity_tensor = torch.zeros(size=(self._num_pairs,), dtype=torch.long)
        # subject head to object head
        head_tensor = torch.zeros(size=(self._num_pairs, self._num_relations), dtype=torch.long)
        tail_tensor = torch.zeros(size=(self._num_pairs, self._num_relations), dtype=torch.long)
        # subject tail to object tail
        for sh, st, rl, oh, ot in sro_set:
            idx = self._m_to_s[(sh, st)]  # 将二元组的索引值转化为 flatten 的索引值
            entity_tensor[idx] = 1  # 默认: sh <= st
            idx = self._m_to_s[(oh, ot)]
            entity_tensor[idx] = 1  # 默认: oh <= ot

            if sh <= oh:
                idx = self._m_to_s[(sh, oh)]
                head_tensor[idx, rl] = 1
            else:
                idx = self._m_to_s[(oh, sh)]
                head_tensor[idx, rl] = 2

            if st <= ot:
                idx = self._m_to_s[(st, ot)]
                tail_tensor[idx, rl] = 1
            else:
                idx = self._m_to_s[(ot, st)]
                tail_tensor[idx, rl] = 2

        return entity_tensor, head_tensor, tail_tensor

    def decode(self, entity_tensor: Tensor, head_tensor: Tensor, tail_tensor: Tensor) -> Set[_SRO_TYPING]:
        """ 将模型预测的结果转化为 `三元组` """

        # ## step1: 提取所有的实体
        entities = {self._s_to_m[s.item()] for s in torch.where(entity_tensor > 0)[0]}

        # ## step2: 构建实体字典, 即根据 head 查询所有可能的 tail
        entity_head_dict = defaultdict(set)
        for eh, et in entities:
            entity_head_dict[eh].add(et)

        # ## step3: 提取所有的 tail 关系, 构成 tail 候选集
        candidate_tails = set()
        for idx, rl in zip(*torch.where(tail_tensor > 0)):
            idx, rl = idx.item(), rl.item()
            if tail_tensor[idx, rl] == 1:  # st 在 ot 前
                st, ot = self._s_to_m[idx]
            else:  # st 在 ot 后
                ot, st = self._s_to_m[idx]
            candidate_tails.add((rl, st, ot))

        # ## step4: 根据 head 关系, 获得结果
        results = set()
        for idx, rl in zip(*torch.where(head_tensor > 0)):
            idx, rl = idx.item(), rl.item()
            if head_tensor[idx, rl] == 1:  # sh 在 oh 前
                sh, oh = self._s_to_m[idx]
            else:  # sh 在 oh 后
                oh, sh = self._s_to_m[idx]
            if sh not in entity_head_dict or oh not in entity_head_dict:
                continue
            # 遍历所有的可能性
            for st in entity_head_dict[sh]:
                for ot in entity_head_dict[oh]:
                    if (rl, st, ot) in candidate_tails:
                        results.add((sh, st, rl, oh, ot))

        return results


if __name__ == '__main__':

    # 实体: (1, 3), (23, 26), (13, 14)
    # head 关系: 22: (1, 23); 23: (1, 13)
    # tail 关系: 22: (2, 26); 23: (3, 14)
    sro_set_ = {
        (1, 3, 22, 23, 26),
        (13, 14, 23, 1, 3),
    }

    scheme_ = TPLinkerScheme(num_relations=24, max_num_tokens=30)
    entity_tensor_, head_tensor_, tail_tensor_ = scheme_.encode(sro_set_)
    print("打印实体的 tensor: ")
    print(entity_tensor_)  # 3 个 1
    print("head 对应关系的 tensor: ")
    print(head_tensor_[:, 22])  # 1 个 1
    print("tail 对应关系的 tensor: ")
    print(tail_tensor_[:, 22])  # 1 个 1
    print("head 对应关系的 tensor: ")
    print(head_tensor_[:, 23])  # 1 个 2
    print("head 对应关系的 tensor: ")
    print(tail_tensor_[:, 23])  # 1 个 2

    print(scheme_.decode(entity_tensor_, head_tensor_, tail_tensor_))
