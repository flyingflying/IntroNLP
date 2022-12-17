# Author: lqxu

import torch
from torch import nn, Tensor

__all__ = ["ConditionalLayerNorm"]


class ConditionalLayerNorm(nn.Module):
    """
    References:
        1. paper: Modulating Language Models with Emotions (https://aclanthology.org/2021.findings-acl.379.pdf)
        2. blog: https://blog.csdn.net/LOVEmy134611/article/details/119114925
        3. document: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """
    def __init__(self, input_size: int, condition_size: int, epsilon: float = 1e-5, condition_hidden_size: int = None):
        super(ConditionalLayerNorm, self).__init__()
        self.epsilon = epsilon

        if condition_hidden_size is not None:
            self.condition_transform = nn.Linear(in_features=condition_size, out_features=condition_hidden_size)
        else:
            self.condition_transform = None
            condition_hidden_size = condition_size
        self.weight_transform = nn.Linear(in_features=condition_hidden_size, out_features=input_size)  # 对应 variance
        self.bias_transform = nn.Linear(in_features=condition_hidden_size, out_features=input_size)  # 对应 mean
        self.reset_parameters()

    def reset_parameters(self):
        # 默认情况下, 转换后的 variance 值是 1
        torch.nn.init.constant_(self.weight_transform.weight, 0)
        torch.nn.init.constant_(self.weight_transform.bias, 1)
        # 默认情况下, 转换后的 mean 值是 0
        torch.nn.init.constant_(self.bias_transform.weight, 0)
        torch.nn.init.constant_(self.bias_transform.bias, 0)

    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        # ## step1: 标准化
        # input 表示的是 "句子", 那么其 shape 应该是 [batch_size, num_words, word_embed_size]
        # layer normalization 本身就是在 word_embed_size 维度上进行归一化
        mean = torch.mean(input, dim=-1, keepdim=True)  # [batch_size, num_words, 1]
        variance = torch.var(input, dim=-1, keepdim=True, unbiased=False)  # [batch_size, num_words, 1]
        std = torch.sqrt(variance + self.epsilon)  # [batch_size, num_words, 1]
        input = (input - mean) / std  # [batch_size, num_words, word_embed_size]

        # ## step2: 反标准化
        # condition 表示的是 "心情", 那么其 shape 应该是 [batch_size, condition_size]
        # 如果神经网络中所有的 LayerNorm 都用同一个 "心情" 嵌入, 那么每一次使用时都要变换一下
        if self.condition_transform is not None:
            condition = self.condition_transform(condition)  # [batch_size, condition_hidden_size]
        weight = torch.unsqueeze(self.weight_transform(condition), dim=1)  # [batch_size, 1, word_embed_size]
        bias = torch.unsqueeze(self.bias_transform(condition), dim=1)      # [batch_size, 1, word_embed_size]
        # 计算 output, 其 shape 应该和 input 是一样的
        output = (input * weight) + bias  # [batch_size, num_words, word_embed_size]

        return output
