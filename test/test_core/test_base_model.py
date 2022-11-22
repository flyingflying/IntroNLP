# Author: lqxu

import shutil

import torch

from core import utils
from core.models import BaseModel, BaseConfig

output_dir = "./output"


if __name__ == '__main__':
    # 初始化模型
    model = BaseModel(BaseConfig(pretrained_name="roberta")).eval()
    # 保存模型
    model.save_pretrained(output_dir)
    # 加载模型
    resume_model = BaseModel.from_pretrained(output_dir).eval()
    resume_model.config.name_or_path = ""  # 保证两个 config 是一致的
    # 删除文件夹
    shutil.rmtree(output_dir)
    # 测试一
    print(model.config)
    print(resume_model.config)
    assert str(model.config) == str(resume_model.config)
    # 测试二
    test_input_ids = torch.randint(low=0, high=20000, size=(5, 10))
    with torch.no_grad():
        result1 = model.bert(test_input_ids, return_dict=True)["last_hidden_state"]
        result2 = resume_model.bert(test_input_ids, return_dict=True)["last_hidden_state"]
    assert utils.is_same_tensor(result1, result2)
