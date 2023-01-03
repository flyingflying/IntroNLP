# Author: lqxu

import _prepare  # noqa

import os

import torch
from torch import nn

from core.utils import get_default_tokenizer

from scheme import GPLinkerEEScheme
from data_modules import argument_labels
from main_train import GPLinkerEESystem, HyperParameters


output_dir = HyperParameters().output_dir

# checkpoint_name = "epoch=129-step=48490.ckpt"
# checkpoint_name = "epoch=59-step=22380.ckpt"

# checkpoint_name = "epoch=49-step=18650.ckpt"

checkpoint_name = "epoch=69-step=26110.ckpt"

tokenizer = get_default_tokenizer()

tokenizer_kwargs = {
    "padding": True, "truncation": True, "max_length": 128,
    "return_attention_mask": False, "return_token_type_ids": False
}


def load_model():
    ckpt_path = os.path.join(output_dir, "checkpoint", checkpoint_name)

    system: GPLinkerEESystem = GPLinkerEESystem.load_from_checkpoint(ckpt_path)

    return system.model.cpu().eval()


@torch.no_grad()
def app(sentence: str, sample=None):

    """ sentence 是句子, sample 是用于 debug 的 """

    # ## 分词
    token_ids = tokenizer(sentence, **tokenizer_kwargs)["input_ids"]

    # ## 加载模型
    model = load_model()
    scheme = GPLinkerEEScheme(argument_labels, ensure_trigger=False)

    # ## 模型预测
    input_ids = torch.tensor([token_ids, ])
    cal_mask = input_ids.ne(0).float()
    token_vectors = model.bert(input_ids, cal_mask)[0]

    arguments_logits = model.argument_classifier(token_vectors)  # [batch_size, num_labels, num_tokens, num_tokens]
    heads_logits = model.head_classifier(token_vectors)
    tails_logits = model.tail_classifier(token_vectors)

    pair_mask = torch.triu(cal_mask.unsqueeze(1) * cal_mask.unsqueeze(-1))  # [batch_size, num_tokens, num_tokens]
    arguments_tensor = (arguments_logits > 0).float() * pair_mask.unsqueeze(1)
    heads_tensor = (heads_logits > 0).float().squeeze(1) * pair_mask
    tails_tensor = (tails_logits > 0).float().squeeze(1) * pair_mask

    if sample is not None:
        # 测试预测的正样本个数
        print("arguments 正样本个数:", torch.sum(arguments_tensor).item())
        print("heads 正样本个数:", torch.sum(heads_tensor).item())
        print("tails 正样本个数:", torch.sum(tails_tensor).item())

        # 测试 loss 值
        gold_arguments_tensor, gold_heads_tensor, gold_tails_tensor = scheme.encode(sample)

        print(torch.where(arguments_logits > 0))
        print(torch.where(heads_logits > 0))
        print(torch.where(tails_logits > 0))

        def multi_label_cross_entropy_loss_with_mask(logits, target, cal_mask_):

            target = target.bool()
            cal_mask_ = cal_mask_.bool().unsqueeze(-1)

            target_logits = torch.where(target & cal_mask_, -logits, -10000.)  # [n_samples, n_labels]
            target_logits = nn.functional.pad(input=target_logits, pad=(0, 1))  # [n_samples, n_labels+1]

            non_target_logits = torch.where((~target) & cal_mask_, logits, -10000.)  # [n_samples, n_labels]
            non_target_logits = nn.functional.pad(input=non_target_logits, pad=(0, 1))  # [n_samples, n_labels+1]

            loss = torch.logsumexp(target_logits, dim=-1) + torch.logsumexp(non_target_logits, dim=-1)  # [n_samples, ]

            print(
                "正类的 loss 值",
                round(
                    (torch.sum(torch.logsumexp(target_logits, dim=-1)) / torch.sum(cal_mask_)).item(),
                    2
                )
            )

            print(
                "负类的 loss 值",
                round(
                    (torch.sum(torch.logsumexp(non_target_logits, dim=-1)) / torch.sum(cal_mask_)).item(),
                    2
                )
            )

            return torch.sum(loss) / torch.sum(cal_mask_)

        # 打印 arguments 的情况

        arguments_loss = multi_label_cross_entropy_loss_with_mask(
            arguments_logits.permute(0, 2, 3, 1),
            gold_arguments_tensor.unsqueeze(0).permute(0, 2, 3, 1),
            pair_mask
        ).item()

        print("arguments 的 loss 值是: ", round(arguments_loss, 2))
        print("arguments 平均 logits 值为: ", torch.mean(arguments_logits).item())
        print("arguments 正类的 logits 值为: ", arguments_logits[gold_arguments_tensor.unsqueeze(0).bool()])

        # 打印 heads 的情况
        heads_loss = multi_label_cross_entropy_loss_with_mask(
            heads_logits.permute(0, 2, 3, 1),
            gold_heads_tensor.unsqueeze(0).unsqueeze(-1),
            pair_mask
        ).item()
        print("heads 的 loss 值是: ", round(heads_loss, 2))
        print("heads 平均 logits 值为: ", torch.mean(heads_logits).item())
        print("heads 正类的 logits 值为: ", heads_logits[gold_heads_tensor.unsqueeze(0).unsqueeze(1).bool()])

        # 打印 tails 的情况
        tails_loss = multi_label_cross_entropy_loss_with_mask(
            tails_logits.permute(0, 2, 3, 1),
            gold_tails_tensor.unsqueeze(0).unsqueeze(-1),
            pair_mask
        ).item()
        print("tails 的 loss 值是: ", round(tails_loss, 2))
        print("tails 平均 logits 值为: ", torch.mean(tails_logits).item())
        print("tails 正类的 logits 值为: ", tails_logits[gold_tails_tensor.unsqueeze(0).unsqueeze(1).bool()])

    # ## 解码
    events = scheme.decode(arguments_tensor[0], heads_tensor[0], tails_tensor[0])

    print("---" * 10)
    for eid, event in enumerate(events):
        event_type = event[0][0]
        print(f"事件{eid}: 类型: {event_type}, 论元有: ")
        for event_type, argument_role, head_idx, tail_idx in event:
            print(argument_role, tokenizer.decode(token_ids[head_idx:tail_idx+1]).replace(" ", ""))
        print("---" * 10)

    return events


if __name__ == '__main__':
    # app("9岁女童失联，租客以婚礼花童为由带走，结果跳江身亡孩子失踪")

    sample_ = {
        "input_ids": [
            101, 7411, 2338, 6161, 1447, 8442, 782, 8038, 3198, 807, 2837, 2461, 872, 3198, 8024, 6825, 2875,
            1461, 6963, 679, 833, 2802, 8013, 102],
        "events": [
            [
                ["组织关系-裁员", "裁员方", 1, 2],
                ["组织关系-裁员", "裁员人数", 5, 6],
                ["组织关系-裁员", "触发词", 3, 4]
            ]
        ]
    }

    app("雀巢裁员4000人：时代抛弃你时，连招呼都不会打！", sample_)
