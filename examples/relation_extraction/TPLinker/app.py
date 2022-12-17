# Author: lqxu

import _prepare  # noqa

import os

import torch

from core.utils import ROOT_DIR
from core.utils import get_default_tokenizer

from data_modules import max_num_tokens, relation_labels, scheme

# global settings
output_dir: str = os.path.join(ROOT_DIR, "examples/relation_extraction/TPLinker/output")
checkpoint_name = "epoch=4-step=51970.ckpt"
# checkpoint_name = "epoch=11-step=124728.ckpt"  # micro: 77.23

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
    from main_train import TPLinkerSystem

    system: TPLinkerSystem = TPLinkerSystem.load_from_checkpoint(_checkpoint_path, map_location="cpu").cpu().eval()

    return system.model


@torch.no_grad()
def app(sentence: str, device: str = "cpu"):
    # 加载模型
    model = load_model().to(device)
    # 准备输入
    input_ids = _tokenizer(sentence, **_tokenizer_kwargs)["input_ids"].to(device)
    # 模型预测
    entity_logits, head_logits, tail_logits = model.forward(input_ids)

    triu_mask = torch.ones(size=(max_num_tokens, max_num_tokens), dtype=torch.bool)
    triu_mask = torch.triu(triu_mask, diagonal=0)
    ignored_mask = (input_ids == 0).unsqueeze(-1).expand(-1, -1, max_num_tokens)[:, triu_mask]

    entity_tensor = entity_logits.argmax(dim=1)
    entity_tensor[ignored_mask] = 0
    head_tensor = head_logits.argmax(dim=1)
    head_tensor[ignored_mask] = 0
    tail_tensor = tail_logits.argmax(dim=1)
    tail_tensor[ignored_mask] = 0

    sro_list = scheme.decode(entity_tensor[0], head_tensor[0], tail_tensor[0])
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
