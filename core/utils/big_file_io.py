# Author: lqxu

"""
文件读取工具包, 暂时只提供大文件行的读取方式, 未来会提供更多的内容
"""

import os
from .file_io import *

__all__ = ["BigFileReader"]


class BigFileReader:
    """
    大文件特定行的读取方式:
        1. 建立行序号和 offset 的对应关系
        2. 根据行序号, 获取 offset, 再根据 offset 读取特定行
    启发: https://www.zhihu.com/question/410819152/answer/1449831259
    TODO: 开发更多的功能, 优化方式, 未来尝试用更好的数据结构来解决, 或者用更底层的语言 (比方说 c++) 来解决 (最好是有第三方包啦 ...)
    """

    def __init__(self, file_path: str, cache_file: str = None):
        if not os.path.exists(file_path):
            raise ValueError(f"{file_path} 文件路径不存在")
        self.file_path = file_path

        if isinstance(cache_file, str) and os.path.exists(cache_file):
            self.offset_mapping = read_json(cache_file)
        else:
            self.offset_mapping = self._build_offset_mapping()
            if cache_file is not None:
                to_json(cache_file, self.offset_mapping, readable=False)

        self._reader = open(file_path, "r", encoding="utf-8")

    def _build_offset_mapping(self):
        offsets = []

        with open(self.file_path, "r", encoding="utf-8") as reader:
            while True:
                offsets.append(reader.tell())  # 记录偏移量
                line = reader.readline()  # 以换行符结尾, 不用担心有多余换行符的问题
                if not line:
                    offsets.pop(-1)  # 一定要后处理, 如果前处理, 会有问题
                    break

        return offsets

    def read_specified_line(self, line_no: int, remove_newlines: bool = True) -> str:
        """ 读取特定行 """
        self._reader.seek(self.offset_mapping[line_no], 0)  # 0 表示从文件开头开始定位
        line = self._reader.readline()
        if remove_newlines:
            line = line.rstrip()
        return line

    def __len__(self): return len(self.offset_mapping)
    def close(self): self._reader.close()
    def __del__(self): self.close()
