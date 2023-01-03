# Author: lqxu

import _prepare  # noqa

from typing import *

import torch
from torch import Tensor

from core.utils import BasicMetrics


class GPLinkerEEMetrics(BasicMetrics):

    def __init__(self, event_labels: List[str], scheme):
        # 这里仅仅进行事件级别的 Precision, Recall 和 F1 score 的计算
        super(GPLinkerEEMetrics, self).__init__(event_labels)

        self.scheme = scheme

    def add(self, reference: List[Any], prediction: List[Tensor], **kwargs):
        arguments_tensor, heads_tensor, tails_tensor = prediction

        pred_events = self.scheme.decode(arguments_tensor, heads_tensor, tails_tensor)

        gold_events = [
            tuple(sorted([tuple(argument) for argument in gold_event]))
            for gold_event in reference if any([argument[1] == "触发词" for argument in gold_event])
        ]

        for pred_event in pred_events:
            if len(pred_event) == 0:
                continue
            event_id = self.labels.index(pred_event[0][0])
            self.counters[event_id].pred_positive += 1

            # if pred_event not in gold_events:
            #     print(sample)
            #     print("解码答案: ")
            #     for event in pred_events:
            #         print(event)
            #     print("问题答案")
            #     print(pred_event)
            #     print("标准答案")
            #     for gold_event in gold_events:
            #         print(gold_event)

        for gold_event in gold_events:
            if len(gold_event) == 0:
                continue
            event_id = self.labels.index(gold_event[0][0])
            self.counters[event_id].gold_positive += 1

            if gold_event in pred_events:
                self.counters[event_id].true_positive += 1

            # if gold_event not in pred_events:
            #     print(sample)
            #     print("标准答案: ")
            #     for event in gold_events:
            #         print(event)
            #     print("问题答案")
            #     print(gold_event)
            #     print("解码答案")
            #     for pred_event in pred_events:
            #         print(pred_event)

    def add_batch(self, references: List[Any], predictions: List[Any], **kwargs):
        arguments_tensor, heads_tensor, tails_tensor = predictions

        for batch_idx in range(len(references)):

            self.add(
                references[batch_idx],
                [arguments_tensor[batch_idx], heads_tensor[batch_idx], tails_tensor[batch_idx]]
            )


class GPLinkerEEAnalysisMetrics(BasicMetrics):

    def __init__(self):
        labels = ["arguments", "head", "tail"]
        super(GPLinkerEEAnalysisMetrics, self).__init__(labels)

    def add(self, reference: Any, prediction: Any, **kwargs):
        self.add_batch(reference, prediction)

    def add_batch(self, references: List[Tensor], predictions: List[Tensor], **kwargs):

        for idx, (pred_tensor, gold_tensor) in enumerate(zip(predictions, references)):

            self.counters[idx].gold_positive += torch.sum(gold_tensor).item()
            self.counters[idx].pred_positive += torch.sum(pred_tensor).item()
            self.counters[idx].true_positive += torch.sum(gold_tensor * pred_tensor).item()


if __name__ == '__main__':

    import os

    from tqdm import tqdm

    from core.utils import ROOT_DIR, read_json_lines

    from data_modules import event_labels as event_labels_
    from data_modules import argument_labels
    from scheme import GPLinkerEEScheme

    output_dir = os.path.join(ROOT_DIR, "examples/event_extraction/GPLinker/output/")

    base_data_dir = os.path.join(output_dir, "base_data")

    scheme_ = GPLinkerEEScheme(argument_labels)
    metrics_ = GPLinkerEEMetrics(event_labels=event_labels_, scheme=scheme_)

    for sample in tqdm(read_json_lines(os.path.join(base_data_dir, "train.jsonl"))):

        model_inputs = scheme_.encode(sample)

        metrics_.add(
            reference=sample["events"],
            prediction=model_inputs
        )

    print(metrics_.classification_report())
