# Author: lqxu

from typing import *

import torch
from torch import Tensor

_MODEL_OUTPUT = Tensor
_SRO_TYPING = Tuple[int, int, int, int, int]


class OneRelScheme:
    def __init__(self, num_relations: int, max_num_tokens: int = 512):
        self.num_relations = num_relations
        self.max_num_tokens = max_num_tokens

    def encode(self, sro_set: Set[_SRO_TYPING]) -> _MODEL_OUTPUT:

        ret = torch.zeros(size=(self.num_relations, self.max_num_tokens, self.max_num_tokens), dtype=torch.long)

        for sh, st, rl, oh, ot in sro_set:
            ret[rl, st, ot] = 3  # 论文中的 HE-TE, 实际上就是 subject-tail, object-tail, 对应 TPLinker 中的 tail_tensor
            ret[rl, sh, ot] = 2  # 论文中的 HB-TE, 实际上就是 subject-head, object-tail, 在 TPLinker 中没有对应
            ret[rl, sh, oh] = 1  # 论文中的 HB-TB, 实际上就是 subject-head, object-head, 对应 TPLinker 中的 head_tensor

        """
        如果 object 是单字实体, 那么 oh = ot, 那么此时 sh-oh 和 sh-ot 位置是相同的, 我们取 sh-oh 的标签
        如果 subject 是单字实体, 那么 sh = st, 那么此时 sh-ot 和 st-ot 位置是相同的, 我们取 sh-ot 的标签
        """

        return ret

    @staticmethod
    def decode_v2(output: _MODEL_OUTPUT, num_tokens: int = None) -> Set[_SRO_TYPING]:

        ret = set()
        num_tokens = output.size(1) if num_tokens is None else num_tokens

        for rl, sh, oh in zip(*torch.where(output == 1)):
            rl, sh, oh = rl.item(), sh.item(), oh.item()
            ot = oh  # 单字 object 的情况

            for candidate_ot in range(oh+1, num_tokens):
                if output[rl, sh, candidate_ot] == 2:
                    ot = candidate_ot
                    break

            st = sh  # 单字 subject 的情况

            for candidate_ht in range(sh+1, num_tokens):
                if output[rl, candidate_ht, ot] == 3:
                    st = candidate_ht
                    break

            ret.add((sh, st, rl, oh, ot))

        return ret

    @staticmethod
    def decode(output: _MODEL_OUTPUT) -> Set[_SRO_TYPING]:

        ret = set()
        # torch.where 返回的结果是有序的, 按照 relations, starts, ends 依次有序, 即
        # 首先是 relations 有序, 在有相同 relations 的情况下, 内部按照 starts 有序, 在有相同 start 的情况下, 内部按照 ends 有序
        relation_labels, subject_heads, objects = torch.where(output > 0)
        num_candidates = len(relation_labels)

        num_tokens = output.size(1)

        for idx, (rl, sh, oh) in enumerate(zip(relation_labels, subject_heads, objects)):
            rl, sh, oh = rl.item(), sh.item(), oh.item()
            if output[rl, sh, oh] != 1:
                continue

            ot = oh  # 单字 object 的情况
            # 利用 torch.where 的特性, 少一次循环, 但是解决不了单个标签实体交叉的情况 (不管用什么方式, 单个标签实体交叉都会有问题的)
            if idx + 1 < num_candidates and output[rl, sh, objects[idx + 1]] == 2:
                ot = objects[idx + 1].item()

            st = sh  # 单字 subject 的情况

            for candidate_ht in range(sh+1, num_tokens):
                if output[rl, candidate_ht, ot] == 3:
                    st = candidate_ht
                    break

            ret.add((sh, st, rl, oh, ot))

        return ret


if __name__ == '__main__':
    scheme_ = OneRelScheme(1, max_num_tokens=30)

    sro_set_ = {
        (1, 3, 0, 23, 26),
        (13, 14, 0, 1, 3),
        (15, 15, 0, 16, 16),
        (17, 17, 0, 17, 17)
    }

    output_ = scheme_.encode(sro_set_)

    for o in output_[0]:
        print(o.tolist())

    print(scheme_.decode_v2(output_))

    print(scheme_.decode(output_))
