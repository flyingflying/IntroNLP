# Author: lqxu

"""
DuReader Retrieval datasets 训练读取 (由于数据量过大, 这里就不使用 pandas 接口了, 直接改用 HuggingFace Datasets 接口) \n
数据集下载地址: https://dataset-bj.cdn.bcebos.com/qianyan/dureader-retrieval-baseline-dataset.tar.gz \n
"""

import os
from typing import *

import pandas as pd
import datasets as hf_datasets

_default_root_data_dir = r"F:\projects2\DuReaderRetrieval\data"


def init_dual_train_data(
        save_path: str, root_data_dir: str = None, debug: bool = True,
        need_tokenized: bool = False, max_q_len: int = 32, max_p_len: int = 384, int_type: str = "int16"):

    root_data_dir = _default_root_data_dir if root_data_dir is None else root_data_dir
    file_name = "train/dual.train.demo.tsv" if debug else "train/dual.train.tsv"
    file_path = os.path.join(root_data_dir, file_name)

    # 数据文件不是很规范, 这里使用 pandas 读取可以解决问题
    dual_train_df = pd.read_csv(
        file_path, sep=r"\t\t", header=None, engine="python", names=["query", "para_pos", "para_neg"])
    hf_dataset = hf_datasets.Dataset.from_pandas(dual_train_df)

    if need_tokenized:
        from datasets.features import Features, Value, Sequence  # typing 中也有 Sequence 对象, 这需要注意一下
        from core.utils import get_default_tokenizer
        tokenizer = get_default_tokenizer()

        tokenizer_kwargs = {
            "padding": "max_length", "truncation": True,
            "return_attention_mask": False, "return_token_type_ids": False
        }

        def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
            return {
                "q_input_ids": tokenizer(
                    batch["query"], **tokenizer_kwargs, max_length=max_q_len)["input_ids"],
                "pp_input_ids": tokenizer(
                    batch["para_pos"], **tokenizer_kwargs, max_length=max_p_len)["input_ids"],
                "np_input_ids": tokenizer(
                    batch["para_neg"], **tokenizer_kwargs, max_length=max_p_len)["input_ids"]
            }

        hf_dataset = hf_dataset.map(
            preprocess, batched=True, remove_columns=["query", "para_pos", "para_neg"],
            features=hf_datasets.Features({
                "q_input_ids": Sequence(Value(int_type)),
                "pp_input_ids": Sequence(Value(int_type)),
                "np_input_ids": Sequence(Value(int_type))
            })
        )
    hf_dataset = hf_dataset.shuffle(seed=3407)
    hf_dataset.save_to_disk(save_path)


def init_cross_train_data(
        save_path: str, root_data_dir: str = None, debug: bool = True,
        need_tokenized: bool = False, max_q_len: int = 32, max_p_len: int = 384, int_type: str = "int16"):

    root_data_dir = _default_root_data_dir if root_data_dir is None else root_data_dir
    file_name = "train/cross.train.demo.tsv" if debug else "train/cross.train.tsv"
    file_path = os.path.join(root_data_dir, file_name)

    cross_train_df = pd.read_csv(
        file_path, sep="\t", header=None, names=["query", "null", "para_text", "label"],
        usecols=["query", "para_text", "label"])
    hf_dataset = hf_datasets.Dataset.from_pandas(cross_train_df)

    if need_tokenized:
        from datasets.features import Features, Value, Sequence  # typing 中也有 Sequence 对象, 这需要注意一下
        from core.utils import get_default_tokenizer
        tokenizer = get_default_tokenizer()

        tokenizer_kwargs = {
            "padding": "max_length", "truncation": True, "max_length": max_q_len + max_p_len + 3,
            "return_attention_mask": False, "return_token_type_ids": False
        }

        def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
            query_text = [sen[:max_q_len] for sen in batch["query"]]
            para_text = [sen[:max_p_len] for sen in batch["para_text"]]
            return {
                "input_ids": tokenizer(
                    text=query_text, text_pair=para_text, **tokenizer_kwargs)["input_ids"],
            }

        hf_dataset = hf_dataset.map(
            preprocess, batched=True, remove_columns=["query", "para_text", ],
            features=hf_datasets.Features({
                "input_ids": Sequence(Value(int_type)),
                "label": Value("int8")
            })
        )
    hf_dataset = hf_dataset.shuffle(seed=3407)
    hf_dataset.save_to_disk(save_path)


if __name__ == '__main__':
    # init_dual_train_data("./outputs", debug=False, need_tokenized=True)
    init_cross_train_data("./outputs", debug=True, need_tokenized=True)
