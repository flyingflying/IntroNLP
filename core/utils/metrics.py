# Author: lqxu

"""
这个模块的设计思路很简单, MetricsResult 类是用来返回数据的, MetricsCounter 类是用来记录数据的;
BasicMetrics 类是用来计算 Precision, Recall, F1 score 的 (有 tp, fp, fn 这些数据就可以计算了);
BasicMetrics 类的子类负责统计 tp, fp 和 fn 这些数据, 通过实现 add 和 add_batch 方法。
"""

from typing import *
from abc import abstractmethod, ABC
from dataclasses import dataclass

__all__ = ["MetricsResult", "BasicMetrics"]


@dataclass
class MetricsResult:
    precision: float  # 预测为正样本的正确率
    recall: float     # 实际为正样本的正确率
    f1_score: float   # precision 和 recall 的调和平均数


@dataclass
class MetricsCounter:
    pred_positive: int = 0  # 预测为正样本数
    gold_positive: int = 0  # 实际为正样本数 (support, weighted macro 中的权重值)
    true_positive: int = 0  # 两者的交集

    def compute(self):
        # ## 如果 true_positive 是 0, 而其它两个数不是 0, 那么意味着统计错误
        if self.true_positive == 0:
            return MetricsResult(precision=0., recall=0., f1_score=0.)
        return MetricsResult(
            precision=self.true_positive / self.pred_positive,
            recall=self.true_positive / self.gold_positive,
            f1_score=(2 * self.true_positive) / (self.gold_positive + self.pred_positive)
        )


class BasicMetrics(ABC):
    labels: List[str]
    counters: List[MetricsCounter]

    @abstractmethod
    def add_batch(self, references: List[Any], predictions: List[Any], **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def add(self, reference: Any, prediction: Any, **kwargs):
        raise NotImplementedError()

    # noinspection PyUnusedLocal
    def __init__(self, labels: List[str], **kwargs):
        assert isinstance(labels, list) and len(labels) > 0 and all(isinstance(label, str) for label in labels)
        self.labels = list(labels)
        self._counter_reset()

    def _clear_invalid_counters(self) -> List[str]:
        labels, counters = [], []
        for label, counter in zip(self.labels, self.counters):
            # invalid counter: 测试集中完全没有出现的样本
            if counter.gold_positive == 0 and counter.pred_positive == 0:
                continue
            counters.append(counter)
            labels.append(label)
        self.counters = counters
        return labels

    def _counter_reset(self): self.counters = [MetricsCounter() for _ in self.labels]

    def _compute_micro(self) -> MetricsResult:
        return MetricsCounter(
            pred_positive=sum(counter.pred_positive for counter in self.counters),
            gold_positive=sum(counter.gold_positive for counter in self.counters),
            true_positive=sum(counter.true_positive for counter in self.counters)
        ).compute()

    @staticmethod
    def _average(numbers: List[float], weights: List[int] = None) -> float:
        if weights is None:
            return sum(numbers) / len(numbers)

        denominator = sum(weights)
        if denominator == 0:
            return 0.
        return sum([weight * number for weight, number in zip(weights, numbers)]) / denominator

    def _compute_macro(self, results: List[MetricsResult]) -> Tuple[MetricsResult, ...]:
        precision = [result.precision for result in results]
        recall = [result.recall for result in results]
        f1_score = [result.f1_score for result in results]
        weights = [counter.gold_positive for counter in self.counters]

        macro_result = MetricsResult(self._average(precision), self._average(recall), self._average(f1_score))
        weighted_macro_result = MetricsResult(
            self._average(precision, weights), self._average(recall, weights), self._average(f1_score, weights))
        return macro_result, weighted_macro_result

    def compute(self, need_reset: bool = True) -> Dict[str, MetricsResult]:
        if need_reset:
            labels = self._clear_invalid_counters()
        else:
            labels = self.labels
        results = [counter.compute() for counter in self.counters]
        ret = {label: result for label, result in zip(labels, results)}
        if len(results) > 1:
            ret["macro"], ret["weighted_macro"] = self._compute_macro(results)
            ret["micro"] = self._compute_micro()
        if need_reset:
            self._counter_reset()
        return ret

    def classification_report(self, digits: int = 4, need_reset: bool = True):
        from seqeval.reporters import StringReporter

        if need_reset:
            labels = self._clear_invalid_counters()
        else:
            labels = self.labels
        label_width = max(map(len, labels))
        avg_width = len('weighted macro avg')
        width = max(label_width, avg_width, digits)
        reporter = StringReporter(width=width, digits=digits)

        results = [counter.compute() for counter in self.counters]
        for label, result, counter in zip(labels, results, self.counters):
            reporter.write(label, result.precision, result.recall, result.f1_score, counter.gold_positive)
        reporter.write_blank()

        if len(results) > 1:
            support = sum(counter.gold_positive for counter in self.counters)
            micro = self._compute_micro()
            reporter.write("micro avg", micro.precision, micro.recall, micro.f1_score, support)
            macro, weighted_macro = self._compute_macro(results)
            reporter.write("macro avg", macro.precision, macro.recall, macro.f1_score, support)
            reporter.write(
                "weighted macro avg", weighted_macro.precision, weighted_macro.recall, weighted_macro.f1_score, support)
            reporter.write_blank()

        if need_reset:
            self._counter_reset()
        return reporter.report()
