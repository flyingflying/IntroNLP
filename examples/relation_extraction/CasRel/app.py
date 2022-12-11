# Author: lqxu

import _prepare  # noqa

import os

import torch

from core.utils import ROOT_DIR
from core.utils import get_default_tokenizer

from data_modules import max_text_len, relation_labels

# global settings
output_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/CasRel/output")
checkpoint_name = "epoch=9-step=34650.ckpt"

# not settings
_tokenizer = get_default_tokenizer()
_checkpoint_path = os.path.join(output_dir, "checkpoint", checkpoint_name)
_tokenizer_kwargs = {
    "max_length": max_text_len, "truncation": True,
    "return_attention_mask": False, "return_token_type_ids": False,
    "return_tensors": "pt",
}


@torch.no_grad()
def load_model():
    from train import CasRelSystem

    system: CasRelSystem = CasRelSystem.load_from_checkpoint(_checkpoint_path, map_location="cpu").cpu().eval()

    return system.model


@torch.no_grad()
def app(sentence: str, device: str = "cpu"):
    # 加载模型
    model = load_model().to(device)
    # 准备输入
    input_ids = _tokenizer(sentence, **_tokenizer_kwargs)["input_ids"].to(device)
    # 模型预测
    sro_list = model.forward(input_ids)
    # 解码
    token_ids = input_ids[0].tolist()
    results = set()
    for sh, st, rl, oh, ot in sro_list:
        # 这里的解码方式是不合理的, 仅仅对中文 ok, 对英文问题很大
        subj = _tokenizer.decode(token_ids[sh:st+1]).replace(" ", "")
        obj = _tokenizer.decode(token_ids[oh:ot+1]).replace(" ", "")
        rel = relation_labels[rl]
        results.add((subj, rel, obj))

    return results


if __name__ == '__main__':
    print(app("王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人"))
