# Author: lqxu

""" 数据处理模块 """

import os
from typing import *

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from core.utils import get_default_tokenizer, read_json_lines, DATA_DIR

from scheme import GPLinkerEEScheme

# 初始化路径
raw_data_dir = os.path.join(DATA_DIR, "event_extraction/DuEE1.0/")

# 初始化分词器
tokenizer = get_default_tokenizer()

tokenizer_kwargs = {
    "padding": "max_length", "truncation": True, "max_length": 128,
    "return_attention_mask": False, "return_token_type_ids": False, "return_offsets_mapping": True
}


def load_argument_labels():
    """ 从 schema 文件中加载所有的论元标签 (总共有 282 个标签) """

    schema_file_path = os.path.join(raw_data_dir, "duee_event_schema.jsonl")
    all_schema = read_json_lines(schema_file_path)

    argument_labels_, event_labels_ = [], []
    for schema in all_schema:
        event_type = schema["event_type"]
        event_labels_.append(event_type)
        argument_labels_.append((event_type, "触发词"))

        for role in schema["role_list"]:
            argument_labels_.append((event_type, role["role"]))

    return argument_labels_, event_labels_


argument_labels, event_labels = load_argument_labels()


def load_data(stage: str = "train"):
    """ 加载数据, 并对文本, 标签进行编码, 样式如下: """

    # language=JSON5
    """
    {
        "input_ids": [  // text 编码后的 ID 值
            101, 679, 788, 788, 3221, 704, 1744, 8233, 821, 689, 1762, 6161, 1447, 8024, 711, 862, 8195, 2487, 4638, 
            4508, 7755, 3152, 738, 1355, 4495, 749, 1059, 4413, 6161, 1447, 102, 
        ], 
        "events": [  // 事件列表
            [  // 一个事件由若干论元组成 (触发词也当作一个论元)
                ["组织关系-裁员", "裁员方", 5, 9],  // 一个论元: (event_type, argument_role, start_idx/head, end_idx/tail)
                ["组织关系-裁员", "触发词", 11, 12]
            ], 
            [
                ["组织关系-裁员", "裁员方", 16, 21], 
                ["组织关系-裁员", "触发词", 28, 29]
            ]
        ]
    }
    """

    # ## step1: 读取数据
    file_name = "duee_train.jsonl" if stage == "train" else "duee_dev.jsonl"
    data_file = os.path.join(raw_data_dir, file_name)
    samples = read_json_lines(data_file)
    new_samples = []

    # ## step2: 分词
    text = [sample["text"] for sample in samples]
    tokenized_results = tokenizer(text, **tokenizer_kwargs)
    all_input_ids = tokenized_results["input_ids"]
    """ offset_mapping 表示的是分词后的每一个 token 在原句中的开始和结束索引, 采用的是 exclusive index \n """
    all_offset_mapping = tokenized_results["offset_mapping"]

    for sample, input_ids, offset_mapping in zip(samples, all_input_ids, all_offset_mapping):

        new_events = []
        for event in sample["event_list"]:  # 遍历样本中的每一个事件
            event_type = event["event_type"]

            """ 每一个事件是由若干论元组成的, 触发词也是论元之一, 将所有的论元变成三元组的形式 """
            arguments = [
                (argument["argument"], argument["argument_start_index"], argument["role"])
                for argument in event["arguments"]
            ]
            arguments.append(
                (event["trigger"], event["trigger_start_index"], "触发词")
            )

            """
            将 offset_mapping 转化成字典:
                对于 start_mapping 来说, key 值是论元在句子中的开始索引值 (0-based), value 是对应的 token 索引值 (0-based)
                对于 end_mapping 来说, key 值是论元在句子中的结束索引值 (exclusive), value 是对应的 token 索引值 (inclusive)
            """
            start_mapping = {
                char_start: idx for idx, (char_start, char_end) in enumerate(offset_mapping) if char_start != char_end
            }
            end_mapping = {
                char_end: idx for idx, (char_start, char_end) in enumerate(offset_mapping) if char_start != char_end
            }

            new_arguments = []
            for argument, start_idx, argument_type in arguments:
                end_idx = start_idx + len(argument)

                # 部分 argument 前面有空格, 需要去除掉
                start_idx += (len(argument) - len(argument.lstrip()))
                # 部分 argument 后面有空格, 需要去除掉 (其实只有一个触发词有这个问题)
                end_idx -= (len(argument) - len(argument.rstrip()))

                if start_idx not in start_mapping or end_idx not in end_mapping:
                    continue

                token_start_idx = start_mapping[start_idx]
                token_end_idx = end_mapping[end_idx]
                new_arguments.append((event_type, argument_type, token_start_idx, token_end_idx))

            new_events.append(new_arguments)

        new_samples.append(
            {"input_ids": input_ids, "events": new_events}
        )

    return new_samples


class DataCollate:
    def __init__(self, is_train_stage: bool = True):
        self.is_train_stage = is_train_stage

        self.scheme = GPLinkerEEScheme(argument_labels, ensure_trigger=True)

    def __call__(self, samples: List[Dict[str, Any]]):
        input_ids = torch.tensor([sample["input_ids"] for sample in samples])  # [batch_size, num_tokens]

        ret = {"input_ids": input_ids}

        if not self.is_train_stage:
            ret["events"] = [sample["events"] for sample in samples]

        arguments_tensor, heads_tensor, tails_tensor = [], [], []

        for sample in samples:
            at, ht, tt = self.scheme.encode(sample)

            arguments_tensor.append(at)
            heads_tensor.append(ht)
            tails_tensor.append(tt)

        arguments_tensor = torch.stack(arguments_tensor, dim=0)  # [batch_size, num_labels, num_tokens, num_tokens]
        heads_tensor = torch.stack(heads_tensor, dim=0)  # [batch_size, num_tokens, num_tokens]
        tails_tensor = torch.stack(tails_tensor, dim=0)  # [batch_size, num_tokens, num_tokens]

        ret.update({"arguments_tensor": arguments_tensor, "heads_tensor": heads_tensor, "tails_tensor": tails_tensor})

        return ret


class DuEEDataModule(LightningDataModule):
    def __init__(self, batch_size: int, output_dir: str, test_over_fitting: bool = False):
        super(DuEEDataModule, self).__init__()

        self.batch_size = batch_size
        self.test_over_fitting = test_over_fitting
        self.base_data_dir = os.path.join(output_dir, "base_data")

        self.scheme = GPLinkerEEScheme(argument_labels, ensure_trigger=True)

        self.train_dataset = self.dev_dataset = self.test_dataset = None

    def prepare_data(self) -> None:
        from core.utils import to_json_lines

        if os.path.exists(self.base_data_dir):
            return

        os.makedirs(self.base_data_dir, exist_ok=True)

        for stage in ["train", "dev"]:
            to_json_lines(
                os.path.join(self.base_data_dir, f"{stage}.jsonl"),
                load_data(stage)
            )

    def setup(self, stage: str) -> None:

        from core.utils import read_json_lines

        if stage == "fit":
            self.train_dataset = read_json_lines(os.path.join(self.base_data_dir, "train.jsonl"))
        if stage in ["fit", "validate", ]:
            self.dev_dataset = read_json_lines(os.path.join(self.base_data_dir, "dev.jsonl"))
        if stage == "test":
            test_file = "train.jsonl" if self.test_over_fitting else "dev.jsonl"
            self.test_dataset = read_json_lines(os.path.join(self.base_data_dir, test_file))
        if stage == "prediction":
            raise NotImplementedError

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


if __name__ == '__main__':

    def show_data():
        import json

        print(json.dumps(load_data("dev")[2], ensure_ascii=False))

    show_data()
