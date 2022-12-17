# Author: lqxu

import os
from typing import *

import torch
from torch import Tensor
from torchtext import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from core.utils import ROOT_DIR
from core.utils import get_default_tokenizer

from scheme import TPLinkerScheme

# 这里借用 CasRel 中的 data_modules 模块数据处理的结果, 仅仅将必要的变量拷贝过来
hf_data_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/CasRel/output", "hf_dataset")

relation_labels = [  # 一共有 48 个关系标签, 考虑到训练难度, 将训练集中数量小于 1000 的标签都删除了, 因此这里只有 34 个标签
    '主演', '作者', '歌手', '导演', '父亲', '成立日期', '妻子', '丈夫', '国籍', '母亲', '作词', '作曲', '毕业院校',
    '所属专辑', '董事长', '朝代', '嘉宾', '出品公司', '编剧', '上映时间', '饰演', '简称', '主持人', '配音', '获奖',
    '主题曲', '校长', '总部地点', '主角', '创始人', '票房', '制片人', '号', '祖籍'
]

max_num_tokens = 192

tokenizer = get_default_tokenizer()

tokenizer_kwargs = {
    "max_length": max_num_tokens, "truncation": True,
    "return_attention_mask": False, "return_token_type_ids": False,
}

scheme = TPLinkerScheme(max_num_tokens=192, num_relations=len(relation_labels))


class DataCollate:
    def __init__(self, is_train_stage: bool):
        self.scheme = scheme
        self.max_num_tokens = max_num_tokens

        self.transforms = transforms.Sequential(
            transforms.ToTensor(padding_value=0),
            transforms.PadTransform(max_num_tokens, pad_value=0)
        )

        triu_mask = torch.ones(size=(max_num_tokens, max_num_tokens), dtype=torch.bool)
        self.triu_mask = torch.triu(triu_mask, diagonal=0)

        self.is_train_stage = is_train_stage

    def __call__(self, batch: List[Dict[str, List[Any]]]) -> Dict[str, Tensor]:
        input_ids = self.transforms([sample["text"] for sample in batch])

        entity_tensor, head_tensor, tail_tensor, sro_sets = [], [], [], []
        for sample in batch:
            sro_set = {tuple(sro) for sro in sample["sro_list"]}
            et, ht, tt = self.scheme.encode(sro_set)

            entity_tensor.append(et)
            head_tensor.append(ht)
            tail_tensor.append(tt)
            sro_sets.append(sro_set)

        entity_tensor = torch.stack(entity_tensor, dim=0)  # [batch_size, num_pairs]
        head_tensor = torch.stack(head_tensor, dim=0)  # [batch_size, num_pairs, num_relations]
        tail_tensor = torch.stack(tail_tensor, dim=0)  # [batch_size, num_pairs, num_relations]

        ignored_mask = (input_ids == 0).unsqueeze(-1).expand(-1, -1, self.max_num_tokens)[:, self.triu_mask]
        entity_tensor[ignored_mask] = -100
        head_tensor[ignored_mask] = -100
        tail_tensor[ignored_mask] = -100

        ret = {
            "input_ids": input_ids, "entity_tensor": entity_tensor,
            "head_tensor": head_tensor, "tail_tensor": tail_tensor
        }
        if not self.is_train_stage:
            ret["ignored_mask"] = ignored_mask
            ret["sro_sets"] = sro_sets

        return ret


class DuIEDataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        super(DuIEDataModule, self).__init__()

        # 超参设置
        self.batch_size = batch_size

        # 其它设置
        self.hf_dataset = None

    def prepare_data(self):
        if not os.path.exists(hf_data_dir):
            raise ValueError("文件路径不存在, 请先调用 CasRel 方法中的 init_hf_dataset 方法 !!!")

    def setup(self, stage: str) -> None:
        from datasets import DatasetDict
        self.hf_dataset = DatasetDict.load_from_disk(hf_data_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.hf_dataset["train"],
            batch_size=self.batch_size, shuffle=True, num_workers=8,
            collate_fn=DataCollate(is_train_stage=True)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.hf_dataset["dev"],
            batch_size=self.batch_size * 4, shuffle=False, num_workers=8,
            collate_fn=DataCollate(is_train_stage=False)
        )

    test_dataloader = val_dataloader
