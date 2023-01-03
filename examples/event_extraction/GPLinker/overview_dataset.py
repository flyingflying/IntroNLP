# Author: lqxu

""" 数据集的基本统计, 以及合理化检测, 以保证数据被正确处理 """

import re
import os

from tqdm import tqdm

from core.utils import read_json_lines
from core.utils import DATA_DIR, ROOT_DIR
from core.utils import get_default_tokenizer

raw_data_dir = os.path.join(DATA_DIR, "event_extraction/DuEE1.0/")

data_dir = os.path.join(ROOT_DIR, "examples/event_extraction/GPLinker/output/data")

max_num_tokens = 128

"""
对于一个 `事件` 来说, 其包含: 事件类型, 事件触发词 (trigger) 和若干 `论元` (argument)

什么是 `论元` 呢? 论元其实就是 NER 中的实体, 就是文本中的一个 `片段` (span), 其是构成 `事件` 的要素之一。
从另一方面来说, 事件抽取就是 NER 的高级版, 将描述同一事件的实体聚在一起。关系抽取和事件抽取其实都是 NER 的进一步应用。

什么是 `触发词` 呢? 在很多时候, 我们很难说某一个文本片段就是一个 `事件`, 只能说这个文本片段描述了一个 `事件`。`事件` 是一个抽象的概念, 但是当
我们看到某些词时, 就知道事件发生了。这样的词被称为 `触发词`。`触发词` 也是句子中的某一个 `片段` (span)。

事件类型就是标签, 是人为定义好需要提取的目标。
"""


def read_data_with_check(stage: str = "train"):
    """ 读取训练 / 测试 数据"""

    # language=JSON
    """
    {
        "text": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！", 
        "event_list": [
            {
                "event_type": "组织关系-裁员", 
                "trigger": "裁员", 
                "trigger_start_index": 2,
                "arguments": [
                    {"argument_start_index": 0, "role": "裁员方", "argument": "雀巢"}, 
                    {"argument_start_index": 4, "role": "裁员人数", "argument": "4000人"}
                ], 
                "class": "组织关系"
            }
        ]
    }  
    """

    file_name = "duee_train.jsonl" if stage == "train" else "duee_dev.jsonl"
    data_file = os.path.join(raw_data_dir, file_name)
    samples = read_json_lines(data_file)

    for sample in samples:
        sample.pop("id")  # 去除 ID 字段
        text = sample["text"]  # 文本

        for event in sample["event_list"]:  # 遍历样本中的每一个事件

            trigger_text = event["trigger"]  # 触发词
            trigger_start_index = event["trigger_start_index"]  # 触发词的起始索引
            trigger_end_index = trigger_start_index + len(trigger_text)  # 触发词的结束索引
            assert text[trigger_start_index:trigger_end_index] == trigger_text

            for argument in event["arguments"]:  # 遍历事件中的每一个论元

                assert len(argument["alias"]) == 0  # 检查所有的 alias 列表是否为空
                argument.pop("alias")  # 去除 alias 字段

                argument_text = argument["argument"]  # 论元文本
                argument_start_index = argument["argument_start_index"]  # 论元文本在句子中的开始索引
                argument_end_index = argument_start_index + len(argument_text)  # 论元文本在句子中的结束索引
                assert text[argument_start_index:argument_end_index] == argument_text

            event.pop("class")

    return samples


def statistics():
    """ 了解数据集的状态 """

    from itertools import chain

    def print_sep1(): print("----" * 20)

    def print_sep2(): print("====" * 20)

    for stage in ["train", "dev"]:
        prefix = "训练集" if stage == "train" else "测试集"

        samples = read_data_with_check(stage)

        # 基本统计
        print(f"{prefix}中总的样本数量为: {len(samples)}")
        print_sep1()

        # 统计文本长度
        text_length = [len(sample["text"]) for sample in samples]
        print(f"{prefix}文本最大长度为: {max(text_length)}")
        print(f"{prefix}文本最小长度为: {min(text_length)}")
        print(f"{prefix}文本平均长度为: {round(sum(text_length) / len(text_length), 2)}")
        print(f"{prefix}文本长度大于 {max_num_tokens} 的数量为: {sum([length > max_num_tokens for length in text_length])}")
        print_sep1()

        # 事件数统计
        event_nums = [len(sample["event_list"]) for sample in samples]
        print(f"{prefix}事件数最多为: {max(event_nums)}")
        print(f"{prefix}事件数最小为: {min(event_nums)}")
        print(f"{prefix}事件数平均为: {round(sum(event_nums) / len(event_nums), 2)}")
        print_sep1()

        # 论元数统计
        """ 存在论元数为 0 的情况, 比方说: 文本中出现了 `触发词` 但是实际事情并没有发生 """
        all_events = list(chain(*[sample["event_list"] for sample in samples]))
        argument_nums = [len(event["arguments"]) for event in all_events]
        print(f"{prefix}论元数最多为: {max(argument_nums)}")
        print(f"{prefix}论元数最小为: {min(argument_nums)}")
        print(f"{prefix}论元数平均为: {round(sum(argument_nums) / len(argument_nums), 2)}")
        print_sep1()

        # 其它统计
        all_arguments = list(chain(*[event["arguments"] for event in all_events]))
        ignored_num = sum([argument["argument_start_index"] > max_num_tokens for argument in all_arguments])
        print(f"{prefix}论元索引位置超过 {max_num_tokens} 的数量为: {ignored_num}")
        print_sep2()


def standardize(text: str):
    """
    标准化文本 \n
    由于数据集中已经标记好了触发词和论元的索引位置, 为了准确性, 分词后的结果是不能多或者少字符的 \n
    因此这里将所有的空白字符都替换成英文逗号, 以确保索引位置的正确性 \n

    BERT 词表中缺少很多中文标签符号, 包括但不限于: 中文双引号, 中文破折号, 中文省略号等等, 由于破折号和省略号没有找到很好的替代品, 这里就不进行替换了
    """
    text = re.sub(r'\s', ',', text).replace('\ue627', ',')
    return text.replace(' ', ',')


def test_tokenizer():
    """ 由于数据中有 trigger_start_index 和 argument_start_index, 这里直接按照字符进行分割 """

    tokenizer = get_default_tokenizer()

    for stage in ["train", "dev"]:
        all_text = [sample["text"] for sample in read_data_with_check(stage)]
        # 将所有的空白字符替换成英文逗号
        all_text_chars = [list(standardize(text)) for text in all_text]

        all_input_ids = tokenizer(
            all_text_chars, add_special_tokens=False, padding=False, truncation=False, is_split_into_words=True,
            return_attention_mask=False, return_token_type_ids=False)["input_ids"]

        for text, input_ids, text_chars in tqdm(zip(all_text, all_input_ids, all_text_chars)):
            if len(text) != len(input_ids):  # or 100 in input_ids:
                print(text)
                print(tokenizer.decode(input_ids))
                print(text_chars)


def test_tokenizer2():
    tokenizer = get_default_tokenizer()

    for stage in ["train", "dev"]:
        samples = read_data_with_check(stage)
        all_text = [sample["text"] for sample in samples]

        results = tokenizer(
            all_text, add_special_tokens=False, padding=False, truncation=False, is_split_into_words=False,
            return_attention_mask=False, return_token_type_ids=False, return_offsets_mapping=True)
        offset_mappings = results["offset_mapping"]

        start_error, end_error = 0, 0

        for offset_mapping, sample in zip(offset_mappings, samples):

            start_mapping = {
                char_start: idx for idx, (char_start, char_end) in enumerate(offset_mapping) if char_start != char_end
            }
            end_mapping = {
                char_end: idx for idx, (char_start, char_end) in enumerate(offset_mapping) if char_start != char_end
            }

            for event in sample["event_list"]:

                arguments = [
                    (argument["argument_start_index"], argument["argument"], argument["role"])
                    for argument in event["arguments"]
                ]
                # 触发词也当作论元
                arguments.append((event["trigger_start_index"], event["trigger"], "触发词"))

                for start_idx, argument_text, argument_type in arguments:
                    end_idx = start_idx + len(argument_text)

                    """ 部分 argument_text 开头有空格 ... """
                    argument_text_ = argument_text.lstrip()
                    erased_length = len(argument_text) - len(argument_text_)
                    start_idx += erased_length
                    argument_text_ = argument_text.rstrip()
                    erased_length = len(argument_text) - len(argument_text_)
                    end_idx -= erased_length

                    if start_idx not in start_mapping:
                        # if not argument_text.startswith(" "):
                        """ 这里的基本都是标记错误 (没有标记完全, 比方说 `10月` 标记成 `0月`), 有 9 个, 直接忽略了 """
                        print(f"start|{argument_text}|{argument_type}")
                        print(sample["text"])
                        print()
                        start_error += 1
                        continue

                    if end_idx not in end_mapping:
                        """ 分词导致的错误, 只有一个, 直接忽略了 """
                        print(f"end|{argument_text}|{argument_type}")
                        print(start_idx, end_idx)
                        print(tokenizer.convert_ids_to_tokens(tokenizer.encode(sample["text"])))
                        print(sample["text"])
                        print()
                        end_error += 1
                        continue

                    start_idx_ = start_mapping[start_idx]
                    end_idx_ = end_mapping[end_idx]  # 这里不需要减 1 !!!!!!
                    assert start_idx_ <= end_idx_

        print(start_error, end_error)


def statistics2():

    """ 按照苏剑林的设计, 一共有 282 个标签, 且有 175 个标签的训练样本数量小于 100, 这训练难度 ...... """

    from prettytable import PrettyTable  # noqa

    schema_file = os.path.join(DATA_DIR, "event_extraction/DuEE1.0/duee_event_schema.jsonl")
    all_schema = read_json_lines(schema_file)
    all_labels = []

    for schema in all_schema:
        event_type = schema["event_type"]
        all_labels.append((event_type, "触发词"))
        for role in schema["role_list"]:
            all_labels.append((event_type, role["role"]))

    print(f"标签总数为: {len(all_labels)}")

    train_label_dict = {label: 0 for label in all_labels}
    dev_label_dict = {label: 0 for label in all_labels}

    for stage in ["train", "dev"]:

        label_dict = train_label_dict if stage == "train" else dev_label_dict

        for sample in read_data_with_check(stage):
            for event in sample["event_list"]:
                event_type = event["event_type"]
                label_dict[(event_type, "触发词")] += 1
                for argument in event["arguments"]:

                    # 长度截取影响不大
                    if argument["argument_start_index"] > max_num_tokens - 10:
                        continue

                    label_dict[(event_type, argument["role"])] += 1

    pt = PrettyTable()  # https://zhuanlan.zhihu.com/p/497233470
    keys = sorted(train_label_dict.keys(), key=train_label_dict.get, reverse=True)
    for key in keys:
        pt.add_row([key, train_label_dict.get(key, 0), dev_label_dict.get(key, 0)])

    pt.field_names = ["标签名", "训练集", "测试集"]
    print(pt.get_string())


if __name__ == '__main__':
    # read_data_with_check()
    # statistics()
    # # test_tokenizer()
    # statistics2()

    test_tokenizer2()
