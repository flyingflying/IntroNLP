# Author: lqxu

from typing import *

from torch import Tensor

from core.utils import BasicMetrics

# 三元组类型: (subj_head, subj_tail, rel_label, obj_head, obj_tail)
_SRO_TYPING = Tuple[int, int, int, int, int]

__all__ = ["SROMetrics", "AnalysisMetrics"]


class SROMetrics(BasicMetrics):

    """ 计算 SRO 的 metrics """

    def add(self, reference: Set[_SRO_TYPING], prediction: Set[_SRO_TYPING], **kwargs):
        for sro in reference:
            self.counters[sro[2]].gold_positive += 1
        for sro in prediction:
            self.counters[sro[2]].pred_positive += 1
            self.counters[sro[2]].true_positive += (1 if sro in reference else 0)

    def add_batch(self, references: List[Any], predictions: List[Any], **kwargs):
        for reference, prediction in zip(references, predictions):
            self.add(reference, prediction)


class AnalysisMetrics(BasicMetrics):

    """ 计算三个预测张量的 metrics, 用于分析哪一部分的效果不好 """

    def __init__(self):
        labels = ["head", "tail", "subject", "object"]

        super(AnalysisMetrics, self).__init__(labels)

        self.mapping = {label: idx for idx, label in enumerate(labels)}

    def add_batch(self, references: Tensor, predictions: Tensor, name: str = "", **kwargs):

        def partial_add(tag_name, r, p):
            self.counters[self.mapping[tag_name]].gold_positive += r.sum()
            self.counters[self.mapping[tag_name]].pred_positive += p.sum()
            self.counters[self.mapping[tag_name]].true_positive += (r * p).sum()

        if name == "entity":
            partial_add("subject", references[:, 0], predictions[:, 0])
            partial_add("object", references[:, 1], predictions[:, 1])
        else:
            partial_add(name, references, predictions)

    add = add_batch
