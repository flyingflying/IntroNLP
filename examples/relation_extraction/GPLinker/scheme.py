# Author: lqxu

import torch

from typing import *

_SRO_TYPING = Tuple[int, int, int, int, int]


class GPLinkerREScheme:
    def __init__(self, num_relations: int, max_num_tokens: int):
        self.num_relations = num_relations

        self.max_num_tokens = max_num_tokens

    def encode(self, sro_set: Set[_SRO_TYPING]):
        """
        对于 TPLinker 来说, 关系识别任务被分成了三部分: 实体识别, 头关系识别 和 尾关系识别 \n
        对于 GPLinker 来说, 头关系识别和尾关系识别和 TPLinker 是一样的, 只是将实体识别拆成了 subject 识别和 object 识别两个任务 \n
        """

        subject_tensor = torch.zeros(self.max_num_tokens, self.max_num_tokens)
        object_tensor = torch.zeros(self.max_num_tokens, self.max_num_tokens)

        head_tensor = torch.zeros(self.num_relations, self.max_num_tokens, self.max_num_tokens)
        tail_tensor = torch.zeros(self.num_relations, self.max_num_tokens, self.max_num_tokens)

        for sh, st, rl, oh, ot in sro_set:
            subject_tensor[sh, st] = 1
            object_tensor[oh, ot] = 1

            head_tensor[rl, sh, oh] = 1
            tail_tensor[rl, st, ot] = 1

        return subject_tensor, object_tensor, head_tensor, tail_tensor

    def decode(self, subject_tensor, object_tensor, head_tensor, tail_tensor):
        ret = set()

        for sh, st in zip(*torch.where(subject_tensor == 1)):
            sh, st = sh.item(), st.item()
            for oh, ot in zip(*torch.where(object_tensor == 1)):
                oh, ot = oh.item(), ot.item()
                for rl in range(self.num_relations):
                    if head_tensor[rl, sh, oh] != 1:
                        continue
                    if tail_tensor[rl, st, ot] != 1:
                        continue
                    ret.add((sh, st, rl, oh, ot))

        return ret


if __name__ == '__main__':

    sro_sets = {
        (1, 3, 22, 23, 26),
        (13, 14, 23, 1, 3)
    }

    scheme = GPLinkerREScheme(25, 64)

    subject_tensor_, object_tensor_, head_tensor_, tail_tensor_ = scheme.encode(sro_sets)

    print(scheme.decode(subject_tensor_, object_tensor_, head_tensor_, tail_tensor_))
