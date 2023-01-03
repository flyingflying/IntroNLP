# Author: lqxu

import torch

from core.trainer.loss_func import multi_label_cross_entropy_loss_with_mask


class MultilabelCategoricalCrossentropy(torch.nn.Module):
    """多标签分类的交叉熵；
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # noqa

    @staticmethod
    def forward(self, y_pred, y_true):
        """
        :param y_true: torch.Tensor, [..., num_classes]
        :param y_pred: torch.Tensor: [..., num_classes]
        """
        y_pred = (1-2*y_true) * y_pred
        y_pred_pos = y_pred - (1-y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


if __name__ == '__main__':

    from core.utils import is_same_tensor

    batch_size, num_tokens, num_labels = 10, 20, 3

    def test_case1():

        logits1 = torch.randn(batch_size, num_tokens, num_tokens, num_labels, requires_grad=True)

        with torch.no_grad():
            logits2 = torch.empty_like(logits1, requires_grad=True)
            logits2.copy_(logits1)

        target = torch.randint(low=0, high=2, size=(batch_size, num_tokens, num_tokens, num_labels)).float()

        mask = torch.ones(batch_size, num_tokens).float()
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        loss1 = multi_label_cross_entropy_loss_with_mask(logits1, target, pair_mask)
        loss2 = MultilabelCategoricalCrossentropy()(logits2, target)
        assert is_same_tensor(loss1, loss2)

        loss1.backward()
        loss2.backward()

        assert is_same_tensor(logits1.grad, logits2.grad)

    def test_case2():

        logits1 = torch.randn(batch_size, num_tokens, num_tokens, num_labels, requires_grad=True)

        with torch.no_grad():
            logits2 = torch.empty_like(logits1, requires_grad=True)
            logits2.copy_(logits1)

        target1 = torch.randint(low=0, high=2, size=(batch_size, num_tokens, num_tokens, num_labels)).float()

        mask = torch.randint(low=0, high=2, size=(batch_size, num_tokens)).float()
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        loss1 = multi_label_cross_entropy_loss_with_mask(logits1, target1, pair_mask)

        logits2 = logits2.masked_fill(
            ~pair_mask.unsqueeze(-1).bool(),
            value=-float('inf')
        )
        """ 对于原版的代码, 其要求被 mask 的地方 target 值一定是 0 """
        with torch.no_grad():
            target2 = torch.empty_like(target1)
            target2.copy_(target1)
            target2[~pair_mask.unsqueeze(-1).expand(-1, -1, -1, num_labels).bool()] = 0

        loss2 = MultilabelCategoricalCrossentropy()(logits2, target2)

        print(loss1)
        print(multi_label_cross_entropy_loss_with_mask(logits1, target2, pair_mask))
        print(loss2)

        # assert is_same_tensor(loss1, loss2)
        #
        # loss1.backward()
        # loss2.backward()
        #
        # assert is_same_tensor(logits1.grad, logits2.grad)

    # for _ in tqdm(range(1000)):
    #     test_case1()

    for _ in range(10):
        test_case2()
        print("-" * 10)
