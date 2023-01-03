# Author: lqxu

""" PRGC 模型对于 DuIE 2.0 数据集的数据处理模块 """

import os
from typing import *

import torch
from tqdm import tqdm
from torch import Tensor
from torchtext import transforms
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict as HFDatasetDict
from pytorch_lightning import LightningDataModule

from core.utils import ROOT_DIR

hf_dataset_path = os.path.join(ROOT_DIR, "examples/relation_extraction/CasRel/output", "hf_dataset")

relation_labels = [  # 一共有 48 个关系标签, 考虑到训练难度, 将训练集中数量小于 1000 的标签都删除了, 因此这里只有 34 个标签
    '主演', '作者', '歌手', '导演', '父亲', '成立日期', '妻子', '丈夫', '国籍', '母亲', '作词', '作曲', '毕业院校',
    '所属专辑', '董事长', '朝代', '嘉宾', '出品公司', '编剧', '上映时间', '饰演', '简称', '主持人', '配音', '获奖',
    '主题曲', '校长', '总部地点', '主角', '创始人', '票房', '制片人', '号', '祖籍'
]

max_num_tokens = 192


class PRGCTrainDataset(Dataset):

    """ 根据 PRGC 模型的特点额外封装了一层 Dataset """

    def __init__(self, hf_dataset: HFDataset):
        instances = []

        for idx, sample in tqdm(enumerate(hf_dataset), desc="生成 PRGC 模型所需要的训练数据"):
            unique_relations = {sro[2] for sro in sample["sro_list"]}
            for relation in unique_relations:
                instances.append((idx, relation))

        self.hf_dataset = hf_dataset
        self.instances = instances

    def __getitem__(self, idx: int):
        sample_idx, selected_relation = self.instances[idx]
        sample = self.hf_dataset[sample_idx]  # 有 text 和 sro_sets 字段
        sample["selected_relation"] = selected_relation
        return sample

    def __len__(self): return len(self.instances)


class DataCollate:
    def __init__(self, is_train_stage: bool):
        self.num_tokens = max_num_tokens
        self.num_relations = len(relation_labels)

        self.transforms = transforms.Sequential(
            transforms.ToTensor(padding_value=0),
            transforms.PadTransform(max_num_tokens, pad_value=0)
        )

        self.is_train_stage = is_train_stage

    def __call__(self, batch: List[Dict[str, List[Any]]]) -> Dict[str, Tensor]:
        input_ids = self.transforms([sample["text"] for sample in batch])

        if not self.is_train_stage:  # 测试阶段

            sro_sets = [{tuple(sro) for sro in sample["sro_list"]} for sample in batch]
            return {"input_ids": input_ids, "sro_sets": sro_sets}

        # 选定的关系: 每一个句子会预选定一个关系去预测 subjects 和 objects
        selected_relations = torch.tensor([sample["selected_relation"] for sample in batch])
        batch_size = input_ids.size(0)

        # 关系预测: 句子级别的多标签分类 (二分类问题)
        relations = torch.zeros(batch_size, self.num_relations)
        # 关联预测: token pairs 级别的二分类问题
        correspondence = torch.zeros(batch_size, self.num_tokens, self.num_tokens)
        # subjects 预测: token 级别的多分类 (BIO 序列标注)
        subjects = torch.zeros(batch_size, self.num_tokens, dtype=torch.long)
        # objects 预测: token 级别的多分类 (BIO 序列标注)
        objects = torch.zeros(batch_size, self.num_tokens, dtype=torch.long)

        for batch_idx, sample in enumerate(batch):
            selected_relation = sample["selected_relation"]
            for sh, st, rl, oh, ot in sample["sro_list"]:
                relations[batch_idx, rl] = 1.0
                correspondence[batch_idx, sh, oh] = 1.0
                if rl != selected_relation:
                    continue

                subjects[batch_idx, sh] = 1  # B-SUBJ
                for idx in range(sh+1, st+1):  # st 是 inclusive 索引, range 需求是 exclusive 索引, 因此要加 1
                    subjects[batch_idx, idx] = 2  # I-SUBJ

                objects[batch_idx, oh] = 1  # B-OBJ
                for idx in range(oh+1, ot+1):
                    objects[batch_idx, idx] = 2  # I-OBJ

        return {
            "input_ids": input_ids, "selected_relations": selected_relations, "relations": relations,
            "correspondence": correspondence, "subjects": subjects, "objects": objects
        }


class DuIE2DataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        super(DuIE2DataModule, self).__init__()
        self.batch_size = batch_size

        self.train_dataset = self.dev_dataset = self.test_dataset = None

    def prepare_data(self) -> None:
        if not os.path.exists(hf_dataset_path):
            raise ValueError("请先调用 CasRel 方法中的 init_hf_dataset 方法")

    def setup(self, stage: str) -> None:
        hf_dataset_dict = HFDatasetDict.load_from_disk(hf_dataset_path)
        if stage == "fit":
            self.train_dataset = PRGCTrainDataset(hf_dataset_dict["train"])

        if stage in ["fit", "validate"]:
            self.dev_dataset = hf_dataset_dict["dev"]

        if stage == "test":
            self.test_dataset = hf_dataset_dict["dev"]

        if stage == "predict":
            raise ValueError("目前没有 predict 阶段的数据集")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
            collate_fn=DataCollate(is_train_stage=True)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dev_dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=8,
            collate_fn=DataCollate(is_train_stage=False)
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=8,
            collate_fn=DataCollate(is_train_stage=False)
        )
