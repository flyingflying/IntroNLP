# Author: lqxu

import _prepare  # noqa

import os

import torch

from core.utils import ROOT_DIR
from core.utils import get_default_tokenizer

from data_modules import max_num_tokens, relation_labels, scheme

# global settings
output_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/OneRel/output")
checkpoint_name = "epoch=1-step=27716.ckpt"

# not settings
_tokenizer = get_default_tokenizer()
_checkpoint_path = os.path.join(output_dir, "checkpoint", checkpoint_name)
_tokenizer_kwargs = {
    "max_length": max_num_tokens, "padding": "max_length", "truncation": True,
    "return_attention_mask": False, "return_token_type_ids": False,
    "return_tensors": "pt",
}


@torch.no_grad()
def load_model():
    from main_train import OneRelSystem

    system: OneRelSystem = OneRelSystem.load_from_checkpoint(_checkpoint_path, map_location="cpu").cpu().eval()

    return system.model


@torch.no_grad()
def app(sentence: str, device: str = "cpu"):
    # 加载模型
    model = load_model().to(device)
    # 准备输入
    input_ids = _tokenizer(sentence, **_tokenizer_kwargs)["input_ids"].to(device)
    # 模型预测
    logits = model.forward(input_ids)  # [batch_size, 4, num_relations, num_tokens, num_tokens]

    result = logits.argmax(dim=1)  # [batch_size, num_relations, num_tokens, num_tokens]
    result = result.permute(0, 2, 3, 1)  # [batch_size, num_tokens, num_tokens, num_relations]
    ignored_mask = (input_ids == 0).unsqueeze(-1).expand(-1, -1, max_num_tokens)  # [batch_size, num_tokens, num_tokens]
    result[ignored_mask] = 0
    result = result.permute(0, 3, 1, 2)  # [batch_size, num_relations, num_tokens, num_tokens]

    sro_list = scheme.decode(result[0])
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
    # (邪少兵王, 作者, 冰火未央)
    print(app("《邪少兵王》是冰火未央写的网络小说连载于旗峰天下"))
