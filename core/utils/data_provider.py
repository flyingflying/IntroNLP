# Author: lqxu
"""
本模块实现的是 pandas 风格的 Dataset 和 collate_fn \n
"""

from typing import *

import pandas as pd
from pandas import DataFrame

import torch
import torchtext
from torch import Tensor
from torch.utils.data import Dataset

from transformers.models.bert import BertTokenizerFast

__all__ = ["DataFrameDataset", "DictDataCollator", "get_default_tokenizer"]


class DataFrameDataset(Dataset):
    def __init__(self, df: DataFrame):
        self.df = df

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 使用 `.iloc` 就好了, 不要使用 `.loc`, 不然要 reset_index
        return self.df.iloc[index, :].to_dict()

    def __len__(self):
        return self.df.shape[0]

    def save_to_disk(self, path: str):
        self.df.to_pickle(path)

    @classmethod
    def load_from_disk(cls, path: str):
        df = pd.read_pickle(path)
        return cls(df)


class DictDataCollator:
    def __init__(self,  text_keys: List[str], other_keys: List[str] = None, max_length: int = None):
        transform_list = [torchtext.transforms.ToTensor(padding_value=0), ]
        if max_length is not None:
            transform_list.insert(0, torchtext.transforms.Truncate(max_length))
            transform_list.append(torchtext.transforms.PadTransform(max_length, pad_value=0))
        self._text_transform = torchtext.transforms.Sequential(*transform_list)

        self._text_keys = text_keys
        self._other_keys = other_keys if other_keys is not None else []

    def __call__(self, samples: List[Any]) -> Dict[str, Tensor]:
        """
        借鉴自 transformers.data.data_collator.torch_default_data_collator \n
        samples 是一个 List, 每一个元素表示一个样本, 可以是 Dict, 可以是一个数据类 \n
        """
        batch = {}

        if not isinstance(samples[0], Mapping):
            samples = [vars(sample) for sample in samples]

        for key in samples[0].keys():
            if key in self._text_keys:
                values = [sample[key] for sample in samples]
                batch[key] = self._text_transform(values)
            elif key in self._other_keys:
                values = [sample[key] for sample in samples]
                batch[key] = torch.tensor(values)

        return batch


_default_tokenizer = None


def get_default_tokenizer() -> BertTokenizerFast:
    global _default_tokenizer

    if _default_tokenizer is None:
        _default_tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext")

    return _default_tokenizer
