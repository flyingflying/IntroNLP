# Author: lqxu

""" CasRel 模型对于 DuIE 2.0 数据集的数据处理模块 """

import os
import random
from typing import *

import torch
from tqdm import tqdm
from torchtext import transforms

from collections import defaultdict
from core.utils import DATA_DIR, read_json_lines, get_default_tokenizer

raw_data_dir = os.path.join(DATA_DIR, "relation_extraction/DuIE2.0")

relation_labels = [  # 一共有 48 个关系标签, 考虑到训练难度, 将训练集中数量小于 1000 的标签都删除了, 因此这里只有 34 个标签
    '主演', '作者', '歌手', '导演', '父亲', '成立日期', '妻子', '丈夫', '国籍', '母亲', '作词', '作曲', '毕业院校',
    '所属专辑', '董事长', '朝代', '嘉宾', '出品公司', '编剧', '上映时间', '饰演', '简称', '主持人', '配音', '获奖',
    '主题曲', '校长', '总部地点', '主角', '创始人', '票房', '制片人', '号', '祖籍'
]

max_text_len = 192

tokenizer = get_default_tokenizer()

tokenizer_kwargs = {
    "max_length": max_text_len, "truncation": True,
    "return_attention_mask": False, "return_token_type_ids": False,
}


def convert_scheme_to_triplet(sample: dict):
    """ 将 DuIE 2.0 的 schema 转换成关系抽取所需要的三元组形式, 转换后的 JSON 格式如下: """
    # language=JSON
    """
    {
        "text": "王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人",
        "sro_list": [
            ["王雪纯", "配音", "晴雯"], 
            ["正大综艺", "主持人", "王雪纯"]
        ]
    }
    """

    text = sample["text"]
    sro_list = []
    for spo in sample["spo_list"]:
        sro_list.append(
            # (subject, relation, object)
            (spo["subject"], spo["predicate"], spo["object"]["@value"])
        )
    return {"text": text, "sro_list": sro_list}


def read_train_data():
    """ 读取训练数据, 并转化成三元组的形式, 返回数据的样式见 `convert_scheme_to_triplet` """
    file_path = os.path.join(raw_data_dir, "duie_train.jsonl")
    samples = read_json_lines(file_path)
    return [convert_scheme_to_triplet(sample) for sample in tqdm(samples, desc="读取训练数据")]


def read_dev_data():
    """ 读取测试数据, 并转化成三元组的形式, 返回数据的样式见 `convert_scheme_to_triplet` """
    file_path = os.path.join(raw_data_dir, "duie_dev.jsonl")
    samples = read_json_lines(file_path)
    return [convert_scheme_to_triplet(sample) for sample in tqdm(samples, desc="读取验证数据")]


def statistics(stage=None):
    """ 对数据集进行简单的数据统计, 确定 max_seq_length 参数和 labels 参数 """

    from pprint import pprint
    from itertools import chain
    from collections import Counter

    def part_statistics(samples):
        # 字符串长度统计
        print("----" * 20)
        text_lens = list(map(len, [sample["text"] for sample in samples]))
        print("text 的最大长度是:", max(text_lens))  # 300 / 299
        print("text 的最小长度是:", min(text_lens))  # 5 / 6
        print("text 的平均长度是:", sum(text_lens) / len(text_lens))  # 69 / 69
        print("text 超过最大长度的个数:", len([text_len for text_len in text_lens if text_len > max_text_len]))  # 2114 / 253

        # 关系数量统计
        print("----" * 20)
        sro_numbers = list(map(len, [sample["sro_list"] for sample in samples]))
        print("sro 数量最多为:", max(sro_numbers))  # 54/26
        print("sro 数量平均为:", sum(sro_numbers) / len(sro_numbers))  # 1.8 / 1.8
        print("sro 数量大于 2 的有:", len([0 for sro_number in sro_numbers if sro_number > 1]))  # 67013 / 8080

        # 统计问题数据数量, 按照论文中所说, 一共有两个问题: EPO 和 SEO
        # EPO 指的是 entity pair overlap, 即实体对重叠
        # SEO 指的是 single entity overlap, 即单个实体重叠
        epo, seo = 0, 0
        for sample in samples:
            entity_pairs = []
            for sro in sample["sro_list"]:  # 获取所有的实体对
                entity_pairs.append((sro[0], sro[2]))  # (subject, object)
                entity_pairs.append((sro[2], sro[0]))  # (object, subject)
            if len(set(entity_pairs)) != len(entity_pairs):
                epo += 1
            entities = []
            for sro in sample["sro_list"]:  # 获取所有的实体
                entities.append(sro[0])
                entities.append(sro[2])
            if len(set(entities)) != len(entities):
                seo += 1
        print("有 `实体对重叠` 问题的样本数量是:", epo)  # 18037 / 2180
        print("有 `实体重叠` 问题的样本数量是:", seo)  # 63309 / 7636

        # 标签数量统计
        print("----" * 20)
        sro_list = chain(*[sample["sro_list"] for sample in samples])
        label_number = Counter([sro[1] for sro in sro_list])
        items = list(label_number.items())
        items.sort(key=lambda item: item[1], reverse=True)
        print("标签数量统计结果如下:")
        pprint(items)
        print("所有的标签如下:", [item[0] for item in items])
        print("----" * 20 + "\n")

    if stage == "train" or stage is None:
        train_data = read_train_data()
        print(f"统计训练数据, 一共有{len(train_data)}个样本")  # 171,135 (17万+)
        part_statistics(train_data)

    if stage == "dev" or stage is None:
        dev_data = read_dev_data()
        print(f"统计测试数据, 一共有{len(dev_data)}个样本")  # 20,652 (2万+)
        part_statistics(dev_data)


def tokenize_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    """
    对 text 字段进行分词, 转化为整型, 同时 sro 三元组转化为:
        [subject 的 head 索引, subject 的 tail 索引, label id 值, object 的 head 索引, object 的 tail 索引]

    注意, 所有的 tail 索引是 inclusive 索引, 这和模型的需求是契合的 (python 支持的索引是 exclusive 索引)

    注意, 这个函数有数据筛选, 不建议使用并行化的计算策略

    数据转化后的样式如下:
    """

    # language=JSON
    """
    {
        "text": [
            101, 4374, 7434, 5283, 3221, 8467, 4276, 517, 5273, 3517, 3457, 518, 704, 3252, 7435, 4638, 
            6981, 7509, 5442, 8024, 1961, 3221, 517, 3633, 1920, 5341, 5686, 518, 4638, 712, 2898, 782, 102
        ], 
        "sro_list": [
            [1, 3, 23, 13, 14], 
            [23, 26, 22, 1, 3]
        ]
    }
    """

    new_samples = []
    text_input_ids = tokenizer([sample["text"] for sample in samples], **tokenizer_kwargs)["input_ids"]

    for text_input_id, sample in tqdm(zip(text_input_ids, samples), desc="分词数据", total=len(text_input_ids)):

        sequence_str = "".join([chr(ele) for ele in text_input_id])
        new_sro_list = []

        for subj, relation_label, obj in sample["sro_list"]:
            if len(subj) == 0 or len(obj) == 0:  # 排除空串的情况 (有空串的情况 !!!)
                continue

            # 编码 label
            if relation_label not in relation_labels:
                continue
            relation_label_id = relation_labels.index(relation_label)

            # 寻找 subject 的位置
            subj = tokenizer.encode(subj, add_special_tokens=False)
            subj_head_index = sequence_str.find("".join([chr(ele) for ele in subj]))
            if subj_head_index == -1:
                continue

            # 寻找 object 的位置
            obj = tokenizer.encode(obj, add_special_tokens=False)
            obj_head_index = sequence_str.find("".join([chr(ele) for ele in obj]))
            if obj_head_index == -1:
                continue

            new_sro_list.append((
                subj_head_index,
                subj_head_index + len(subj) - 1,  # inclusive index
                relation_label_id,
                obj_head_index,
                obj_head_index + len(obj) - 1  # inclusive index
            ))

        # 数据筛选
        if len(new_sro_list) == 0:  # 没有 SRO (可能由于标签筛选原因, 可能由于截断原因, 可能由于分词原因 ...), 都过滤掉
            continue

        new_samples.append({
            "text": text_input_id, "sro_list": new_sro_list
        })

    return new_samples


def init_hf_dataset(save_dir: str, debug: bool = False):
    from datasets import Dataset, DatasetDict
    from datasets.features import Features, Sequence, Value

    features = Features({
        "text": Sequence(Value(dtype="uint16")),
        "sro_list": Sequence(Sequence(Value(dtype="uint8")))
    })

    train_samples = tokenize_samples(read_train_data())
    dev_samples = tokenize_samples(read_dev_data())

    hf_dataset = DatasetDict({
        "train": Dataset.from_list(train_samples, features=features),
        "dev": Dataset.from_list(dev_samples, features=features)
    })

    if debug:
        print(hf_dataset)
    else:
        hf_dataset.shuffle(seed=3407)
        hf_dataset.save_to_disk(save_dir)


class TrainCollateFn:
    def __init__(self):
        self.transforms = transforms.Sequential(
            transforms.ToTensor(padding_value=0),
            transforms.PadTransform(max_text_len, pad_value=0)
        )

    def __call__(self, batch: List[Dict[str, List[Any]]]) -> Dict[str, torch.Tensor]:

        """ 准备数据 (这里的数据不适合存硬盘中, 太大了) """

        input_ids = self.transforms([sample["text"] for sample in batch])
        batch_size, num_tokens = input_ids.shape

        span_subj_heads = torch.zeros(batch_size, num_tokens, 1)
        span_subj_tails = torch.zeros(batch_size, num_tokens, 1)

        selected_subj_head = []  # "整型列表" 索引
        selected_subj_tail = []  # "整型列表" 索引

        span_obj_heads = torch.zeros(batch_size, num_tokens, len(relation_labels))
        span_obj_tails = torch.zeros(batch_size, num_tokens, len(relation_labels))

        for batch_idx, sample in enumerate(batch):
            s2ro_map = defaultdict(list)
            for subj_head, subj_tail, label_idx, obj_head, obj_tail in sample["sro_list"]:
                # 模型首先识别 subject, 是基于 span 的, 因此直接是两个数组
                span_subj_heads[batch_idx, subj_head, 0] = 1.
                span_subj_tails[batch_idx, subj_tail, 0] = 1.
                # 一个 subject 可能有多个关系, 因此需要首先构造字典
                s2ro_map[(subj_head, subj_tail)].append((label_idx, obj_head, obj_tail))

            # 随机选择一个 subject 去预测 (不用全部是因为每一个样本的关系数不是一个定值, 无法并行化计算)
            subj_head, subj_tail = random.choice(list(s2ro_map.keys()))
            selected_subj_head.append(subj_head)
            selected_subj_tail.append(subj_tail)

            for label_idx, obj_head, obj_tail in s2ro_map[(subj_head, subj_tail)]:
                span_obj_heads[batch_idx, obj_head, label_idx] = 1.
                span_obj_tails[batch_idx, obj_tail, label_idx] = 1.

        return {
            "input_ids": input_ids, "span_subj_heads": span_subj_heads, "span_subj_tails": span_subj_tails,
            "selected_subj_head": selected_subj_head, "selected_subj_tail": selected_subj_tail,
            "span_obj_heads": span_obj_heads, "span_obj_tails": span_obj_tails
        }


def test_collate_fn(batch: List[Dict[str, List[Any]]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("uncorrected batch size !!!")

    sample = batch[0]
    input_ids = torch.tensor([sample["text"]])  # [1, num_tokens]
    sro_set = {tuple(sro) for sro in sample["sro_list"]}

    return {
        "input_ids": input_ids, "sro_set": sro_set
    }


def check():
    """ 用于检查 collate_fn 函数的正确性, 避免不必要的麻烦 """
    samples = [{
        "text": "王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人",
        "sro_list": [("王雪纯", "配音", "晴雯"), ("王雪纯", "主持人", "正大综艺")]  # 这里故意将 正大综艺 和 王雪纯 反过来的
    }]

    samples = tokenize_samples(samples)
    print(samples[0]["text"])
    print(samples[0]["sro_list"])
    batch = TrainCollateFn()(samples)

    for key in ["input_ids", "span_subj_heads", "span_subj_tails", ]:
        print(f"\n{key}: ")
        print(batch[key][0].tolist())

    for key in ["selected_subj_head", "selected_subj_tail"]:
        print(f"\n{key}: ")
        print(batch[key])

    for key in ["span_obj_heads", "span_obj_tails"]:
        print(f"\n{key}: ")
        for i, t in enumerate(batch[key][0]):
            print(i, t.tolist())

    print(test_collate_fn(samples))


if __name__ == '__main__':
    # 了解数据集的状况
    # statistics()
    # 检查代码的正确性
    check()
    # init_hf_dataset("", debug=True)
