# Author: lqxu
# 文件读写相关

import json
import pickle
from typing import *
from enum import Enum
from datetime import datetime
from dataclasses import asdict, is_dataclass


__all__ = [
    "to_json", "read_json",
    "to_json_lines", "read_json_lines",
    "to_pickle", "read_pickle"
]


def read_file_by_lines(file_name: str, encoding: str = "utf-8", remove_newlines=True):
    with open(file_name, "r", encoding=encoding) as reader:
        lines = reader.readlines()
    if remove_newlines:
        lines = [line.rstrip() for line in lines]  # 去除末尾的空格
    return lines


def read_file_by_iterator(file_name: str, encoding: str = "utf-8", remove_newlines=True):
    with open(file_name, "r", encoding=encoding) as reader:
        for line in reader:
            if remove_newlines:
                yield line.strip()
            else:
                yield line


def write_file_by_lines(file_name: str, lines: Iterable, encoding: str = "utf-8", need_newlines=True):
    if need_newlines:
        lines = [line + "\n" for line in lines]
    with open(file_name, "w", encoding=encoding) as writer:
        writer.writelines(lines)


def to_json(file_name: str, obj: Any, readable: bool = True):
    # json serializable: https://docs.python.org/3/library/json.html#json.JSONEncoder
    # 这里仅仅支持文件的序列化, 不要用于 web 开发, web 开发应该使用框架自带的序列化方式
    if is_dataclass(obj):
        obj = asdict(obj)
    elif hasattr(obj, "__dict__"):  # dict, list, tuple 等 build-in 类是没有 __dict__ 属性的
        obj = vars(obj)
    assert isinstance(obj, (dict, list, tuple))  # int, float, bool, None 虽然也没问题, 但是意义不大

    kwargs = {"ensure_ascii": False}
    if readable:
        kwargs.update({"indent": 4, "sort_keys": True})
        try:
            json.dumps(obj, **kwargs)  # 测试一下
        except TypeError:
            # 如果键的类型不统一, 比方说有 str 和 int, 那么会报错, 因为它们比较不了, 此时只能将 sort_keys 删除
            kwargs.pop("sort_keys")

    with open(file_name, "w", encoding="utf-8") as writer:
        json.dump(obj, writer, **kwargs)


def to_json_lines(file_name: str, objects: Iterable):
    lines = []
    for object_ in objects:
        if is_dataclass(object_): object_ = asdict(object_)  # noqa: E701
        elif hasattr(object_, "__dict__"): object_ = vars(object_)  # noqa: E701

        lines.append(json.dumps(object_, ensure_ascii=False))
    write_file_by_lines(file_name, lines, encoding="utf-8", need_newlines=True)


def read_json(file_name: str, func: Callable = None):
    """ 读取 json 文件, 对于大文件读取, 建议使用 pandas.read_json """
    with open(file_name, "r", encoding="utf-8") as reader:
        obj = json.load(reader)
    if isinstance(func, Callable) and isinstance(obj, dict):
        obj = func(**obj)
    return obj


def read_json_lines(file_name: str, func: Callable = None) -> List:
    """ 读取 json 文件, 对于大文件读取, 建议使用 pandas.read_json """
    with open(file_name, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    lines = [json.loads(line) for line in lines if line]
    if isinstance(func, Callable) and isinstance(lines[0], dict):
        lines = [func(**line) for line in lines]
    return lines


def to_pickle(file_name: str, obj: Any, protocol: int = None):
    # protocol 相关的: https://docs.python.org/3/library/pickle.html?highlight=protocol#data-stream-format
    assert protocol is None or (isinstance(protocol, int) and 0 <= protocol <= pickle.HIGHEST_PROTOCOL)
    with open(file_name, "wb") as writer:
        pickle.dump(obj, writer, protocol=protocol)


def read_pickle(file_name: str):
    with open(file_name, "rb") as reader:
        obj = pickle.load(reader)
    return obj


# TODO: 这个方法需要更多的测试和优化, 比方说 由于使用了递归, 需要考虑设置最大调用数等等
def to_json_advance(
        file_name: str, obj: Any,
        readable: bool = True,
        datetime_fmt: str = '%Y-%m-%d %H:%M:%S'
):
    def recursive(object_, ancestor_object_ids: Set[int]):
        if isinstance(object_, Enum):  # 这一步一定要在前面
            object_ = object_.value  # 对于 Enum 对象, 直接取其 value 值

        if isinstance(object_, datetime):
            return object_.strftime(datetime_fmt) if datetime_fmt is not None else str(object_)
        if isinstance(object_, (str, int, float, bool)) or object_ is None:  # id 重复没关系, 并且对于同一个对象都是单例模式
            return object_

        child_object_ids = set(ancestor_object_ids)

        def check_circular(c_object):
            object_id = id(c_object)
            assert object_id not in ancestor_object_ids, "circular happened!!!"
            child_object_ids.add(object_id)

        if is_dataclass(object_):
            check_circular(object_)
            object_ = asdict(object_)

        if isinstance(object_, dict):
            check_circular(object_)
            ret = {}
            for key, value in object_.items():
                # json.dumps 中会有类似的检查
                assert isinstance(key, (str, int, float, bool)) or key is None, \
                    f"keys must be str, int, float, bool or None, not {type(key)}"
                ret[key] = recursive(value, child_object_ids)
            return ret

        if isinstance(object_, Iterable):  # 子 id 不能包含父 id
            check_circular(object_)
            return [recursive(object_item, child_object_ids) for object_item in object_]

        if hasattr(object_, "__str__"):
            return str(object_)

        raise AssertionError(f"Object of type {type(object_)} is not JSON serializable")

        # list, tuple, dict, datetime 类型都是没有 __dict__ 属性的, 但是这样写还是很危险的
        # if hasattr(object_, "__dict__"):
        #     object_id = id(object_)
        #     assert object_id not in id_objects, "circular happened!!!"
        #     id_objects.add(object_id)
        #     object_ = asdict(object_) if is_dataclass(object_) else vars(object_)

    to_json(file_name, recursive(obj, set()), readable=readable)
