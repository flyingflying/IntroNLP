# Author: lqxu

""" 本模块仅供学习使用 """

import torch
from torch import Tensor, LongTensor
from torch.nn import functional as F  # noqa

__all__ = ["nll_loss", "cross_entropy", "binary_cross_entropy", "binary_cross_entropy_with_logits"]


def nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    # type: (Tensor, LongTensor, Tensor, int, str, float) -> Tensor
    """ negative log likelihood loss \n
    这里虽然名字是 negative log likelihood loss, 但是实际上实现的 negative loss. \n
    虽然这个名字很坑, 但是如果你自己实现 loss 的话, 这个函数很有用的 \n
    input 的 shape 是 [batch_size, num_classes, other_dims], 其中 batch_size 和 other_dims 可以没有,
    other_dims 的个数也可以是多个, 没有限制也就是说:
        1. 如果 ndim=1, 那么 num_classes 在第一个维度
        2. 如果 ndim>1, 那么无论什么情况下, num_classes 都在第二个维度
    target 的 shape 是 [batch_size, other_dims], 其和 input 相比就是去掉了 num_classes 维度 \n
    从本质上说, 其样本数是 batch_size * other_dims, 不支持多个分类任务 \n
    weight 的 shape 是 [num_classes, ], 用于平衡不同的类别的 \n
    如果 reduction 是 "mean" 的话, 求的是加权平均数, 每一个 "样本" 的权重等于其目标类的 weight \n
    这个函数在 python 层面的实现是很难的, 这里是不考虑性能的强制实现, 如果你自己实现 loss, 能使用这个函数建议直接使用 \n
    原版是没有 label_smoothing, 但是我个人觉得在这里实现更加合理 \n
    references:
        1. https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        2. https://zhuanlan.zhihu.com/p/99256429
        3. https://discuss.pytorch.org/t/using-nllloss-weighted-loss-how-to-define-a-good-loss/39263
    """

    if input.ndim == 0:
        raise ValueError("input 张量至少应该有一个维度")
    if input.ndim != target.ndim + 1:
        raise ValueError("target 张量的维度应该比 input 少一个")

    # ## loss_part1 是每一个样本对应每一个类别的 loss 值
    if input.ndim == 1:
        num_classes = input.size(0)
        loss_part1 = -input
    else:
        num_classes = input.size(1)
        loss_part1 = -input.movedim(source=1, destination=-1)

    if loss_part1.shape[:-1] != target.shape:
        raise ValueError("input 张量的 shape 和 target 张量的 shape 不一致")

    if weight is None:
        weight = torch.ones(size=(num_classes, ))
    elif weight.shape != (num_classes, ):
        raise ValueError("weight 张量的 shape 应该是: (num_classes, )")

    # ## loss_part2 是每一个样本实际的类别概率分布
    new_target = torch.zeros_like(target, dtype=torch.int64).copy_(target)
    new_target[target == ignore_index] = num_classes
    loss_part2 = F.one_hot(new_target, num_classes=num_classes+1).float()
    loss_part2 = loss_part2[..., :-1]
    if label_smoothing != 0.0:
        # 目标类的概率为: (1 - label_smoothing) + (label_smoothing / num_classes)
        # 非目标类的概率为: label_smoothing / num_classes
        loss_part2 = loss_part2 * (1 - label_smoothing) + (label_smoothing / num_classes)
        # 确保 ignore_index 处都是 0
        loss_part2 = (target != ignore_index).unsqueeze(-1).float() * loss_part2

    # loss_part1 和 loss_part2 的 shape 都是 [(batch_size, other_dims,) num_classes]
    # weight 的 shape 是 [num_classes, ]
    loss = torch.sum(loss_part1 * loss_part2 * weight, dim=-1)

    if reduction == "mean":
        # ## sample_weight 是每一个样本的权重值, 即其目标类的权重值
        sample_weight = weight[target[target != ignore_index]]
        return torch.sum(loss / torch.sum(sample_weight))
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"invalid reduction method of {reduction}")


def cross_entropy(input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    # type: (Tensor, Tensor, Tensor, int, str, float) -> Tensor
    """ cross entropy loss \n
    多分类问题的交叉熵 loss, 信息熵的计算使用 F.log_softmax 函数即可, 重点关注 label_smoothing, 注意这里和论文中的公式不完全一样:
        + 目标类的概率为: (1 - label_smoothing) + (label_smoothing / num_classes)
        + 非目标类的概率为: label_smoothing / num_classes
    另外需要注意的是 weight + label_smoothing 时的计算方式 \n
    parameters' shape:
        + input: [(batch_size), num_classes, (other_dims, ...)]
        + target: [(batch_size, other_dims, ...)]
        + weight: [num_classes, ]
    reference:
        1. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        2. https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/LossNLL.cpp
    """

    if input.ndim == 0:
        raise ValueError("input 张量至少应该有一个维度")
    num_classes_dim = 0 if input.ndim == 1 else 1

    log_probs = F.log_softmax(input, dim=num_classes_dim)  # [batch_size, num_classes, ...]
    num_classes = input.size(num_classes_dim)

    loss_part2 = F.nll_loss(  # [batch_size, ...]
        input=log_probs, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)

    if label_smoothing == 0.0:
        return loss_part2

    if weight is None:  # 不需要对 weight 做额外的参数检查了, 因为 nll_loss 中已经进行过了
        weight = torch.ones(size=(num_classes, ))

    # 每一类的权重值等于其本身的权重值, 不是目标类的权重值
    log_probs_with_weight = log_probs.movedim(source=num_classes_dim, destination=-1) * weight
    loss_part1 = -torch.sum(log_probs_with_weight, dim=-1)  # [batch_size, ...]
    loss_part1[target == ignore_index] = 0.

    if reduction == "mean":
        """
        1. 这一步使用了两次 `__getitem__` 方法, 但是含义是不一样的, 分别对应: torch.gather 和 torch.masked_selected 方法
            1.1 target[target != ignore_index] 返回的是一个一维张量, 相当于在使用 `torch.masked_selected` 方法
            1.2 weight[target] 返回的依然是一个一维张量, 由于 target 是 LongTensor, 返回的和 target 的 shape 是一致的,
                相当于在使用 `torch.gather` 方法
        2. 注意在有 weight 参数的情况下, "mean" 表示的是对每一个样本的 loss 求加权平均数, 权重是样本实际目标类的权重值
        """
        # 错误的写法: sample_weight = weight.clone()[target]; sample_weight[target == ignore_index] = 0.
        sample_weight = weight[target[target != ignore_index]]
        # sample_weight = torch.gather(
        #     input=weight, dim=0,
        #     index=torch.masked_select(input=target, mask=(target != ignore_index))
        # )
        loss_part1 = torch.sum(loss_part1) / torch.sum(sample_weight)
    elif reduction == "sum":
        loss_part1 = torch.sum(loss_part1)
    elif reduction != "none":
        raise ValueError(f"invalid reduction method of {reduction}")

    loss = (label_smoothing / num_classes) * loss_part1 + (1 - label_smoothing) * loss_part2
    return loss


def binary_cross_entropy(input: Tensor, target: Tensor, weight: Tensor = None, reduction: str = "mean") -> Tensor:
    """ binary cross entropy \n
    二分类的交叉熵, input 和 target 的 shape 一致即可, 没有任何限制; weight 的 shape 只要能和 input 张量进行 element-wise 的运算即可 \n
    为了方便说明, 我们后面设定 input 和 target 的 shape 是 [batch_size, num_classes], weight 的 shape 是 [num_classes, ] \n
    本质上来说, 就是进行 batch_size * num_classes 个 "二分类", 天然支持多任务分类 (上面的 CrossEntropy 不支持多任务!!!) \n
    这里的 weight 不是 "类别" 的权重, 而是 "任务" 的权重, 这和上面的 CrossEntropy 也不一样 !!! \n
    另外需要说明的是 target 是 FloatTensor, 不是 LongTensor, 理论上必须在 [0, 1] 之间, 用其可以实现 label_smoothing, 即:
        + 正类: 1 - label_smoothing / 2
        + 负类: label_smoothing / 2
    reference:
        1. https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """

    if input.shape != target.shape:
        raise ValueError("input 和 target 的 shape 应该一致")
    target = target.float()
    loss = -(target * torch.log(input) + (1 - target) * torch.log(1 - input))
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss  # 注意这里返回的是 [batch_size, num_classes], 不是 [batch_size, ]
    raise ValueError(f"invalid reduction method of {reduction}")


def binary_cross_entropy_with_logits(input, target, weight=None, pos_weight=None, reduction="mean"):
    # type: (Tensor, Tensor, Tensor, Tensor, str) -> Tensor
    """ binary cross entropy with sigmoid layer \n
    和 binary_cross_entropy 一样, 只是 input 没有经过 sigmoid 运算 \n
    多了个 pos_weight, 其表示正类的权重值, 用于 weighted cross entropy, shape 的要求和 weight 一样, 能和 input 进行 element-wise 运算即可 \n
    reference: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """
    if input.shape != target.shape:
        raise ValueError("input 和 target 的 shape 应该一致")
    target = target.float()

    log_probs = F.logsigmoid(input)
    pos_loss = target * log_probs
    if pos_weight is not None:
        pos_loss = pos_loss * pos_weight
    neg_loss = (1 - target) * torch.log(1 - torch.exp(log_probs))
    loss = -(pos_loss + neg_loss)
    if weight is not None:
        loss = loss * weight

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        # 注意这里不需要算加权平均数, 直接算平均数即可
        return loss.mean()
    if reduction == "none":
        return loss  # 注意这里返回的是 [batch_size, num_classes], 不是 [batch_size, ]
    raise ValueError(f"invalid reduction method of {reduction}")


# noinspection PyRedeclaration
nll_loss = F.nll_loss
# noinspection PyRedeclaration
cross_entropy = F.cross_entropy
# noinspection PyRedeclaration
binary_cross_entropy = F.binary_cross_entropy
# noinspection PyRedeclaration
binary_cross_entropy_with_logits = F.binary_cross_entropy_with_logits
