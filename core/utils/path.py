# Author: lqxu

"""
项目的根目录, 可以在项目文件夹中的任意位置启动 Python 程序, 移植到其它项目中时只要修改 `__project_name` 变量即可。\n
未来考虑设置环境变量, 让所有的项目都可以共享资源。
"""

import os

__project_name = "IntroNLP"

__all__ = ["ROOT_DIR", "DATA_DIR", ]


def __get_root_dir(project_name: str) -> str:
    root_dir = os.getcwd()
    while True:
        base_name = os.path.basename(root_dir)
        if base_name == project_name:
            return root_dir
        if base_name == "":
            return os.getcwd()
        root_dir = os.path.dirname(root_dir)


def __get_resources_dir(root_dir: str):
    return os.path.normpath(
        os.path.join(root_dir, "datasets")
    )


ROOT_DIR = __get_root_dir(__project_name)
DATA_DIR = __get_resources_dir(ROOT_DIR)
