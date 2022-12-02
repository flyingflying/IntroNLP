# Author: lqxu

import _prepare  # noqa

import os
from typing import *

from core.utils import read_json_lines, ROOT_DIR, read_json

phase2_root_dir = os.path.join(ROOT_DIR, "./examples/RocketQA/02/")
raw_data_dir: str = os.path.join(ROOT_DIR, "./datasets/du_reader_retrieval/")


def load_reference() -> Dict[str, List[str]]:
    """ 加载标准答案 """
    ret = {}
    reference_path = os.path.join(raw_data_dir, "dev/dev.json")
    dev_samples = read_json_lines(reference_path)

    for dev_sample in dev_samples:
        q_id = dev_sample["question_id"]
        p_id_list = [answer_paragraph["paragraph_id"] for answer_paragraph in dev_sample["answer_paragraphs"]]
        ret[q_id] = p_id_list

    return ret


def load_candidate() -> Dict[str, List[str]]:
    """ 加载候选集 """
    candidate_path = os.path.join(phase2_root_dir, "dev_results.json")
    ret = read_json(candidate_path)
    return ret


def compute_metrics(candidates: Dict[str, List[str]], references: Dict[str, List[str]]) -> Dict[str, float]:
    """
    计算 MRR (Mean Reciprocal Rank)

    recall@50: "正确" 的定义: 在前 50 的预测文本中, 有文本在 ”标准答案“ 中出现 \n
    MRR@50: "正确" 的定义和上面是一致的, 只是不再计算正确率, 而是将 ”正确“ 答案的索引 (1-based) 取倒数然后相加, 如果样本 "错误", 则记作 0 \n

    按照 RocketQA 中的代码, 这里会计算 MRR@10, recall@1 和 recall@50

    Reference:
        https://blog.csdn.net/jiangjiang_jian/article/details/108246103 \n
        https://github.com/PaddlePaddle/RocketQA/blob/main/research/DuReader-Retrieval-Baseline/metric/evaluation.py \n
    """
    mrr_10, recall_1, recall_50 = 0., 0., 0.
    for q_id, candidate_list in candidates.items():
        # 假设 candidates 中的文本在 references 中都有 (如果没有的话, 会导致测评结果值偏低)
        reference_list = references[q_id]

        for i, p_id in enumerate(candidate_list):
            if p_id in reference_list:
                if i < 1:
                    recall_1 += 1.
                if i < 10:
                    mrr_10 += 1. / (i + 1)
                if i < 50:
                    recall_50 += 1.
                break

    return {
        "MRR@10": mrr_10 / len(candidates),
        "recall@1": recall_1 / len(candidates),
        "recall@50": recall_50 / len(candidates),
    }


if __name__ == '__main__':
    from pprint import pprint

    pprint(compute_metrics(load_candidate(), load_reference()))
