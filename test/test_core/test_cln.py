# Author: lqxu

from typing import *
from operator import mul
from functools import reduce
from unittest import TestCase

import torch
from torch import nn, Tensor

from core.modules import ConditionalLayerNorm
from core.utils import is_same_tensor


class NativeLayerNorm(nn.Module):
    """ 这个类对应 torch.nn.LayerNorm 类, 是用来学习的, 不是用来使用的 """
    def __init__(self, normalized_shape: Tuple[int, ...], eps: float = 1e-5, elementwise_affine: bool = True,):
        super(NativeLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(size=normalized_shape))
            self.bias = nn.Parameter(torch.empty(size=normalized_shape))
        else:
            self.weight = self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.constant_(self.weight, 1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return self.layer_norm(
            input=input, normalized_shape=self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)

    @staticmethod
    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        # type: (Tensor, Tuple[int, ...], Tensor, Tensor, float) -> Tensor
        """ 这个方法对应 torch.layer_norm 方法 """
        # ## step1: 修改 input 张量的 shape
        input_shape = input.shape
        assert input_shape[len(normalized_shape):] == normalized_shape
        normalized_size = reduce(mul, normalized_shape)
        input = input.view(-1, normalized_size)

        # ## step2: 标准化: 均值为 0, 方差为 1
        mean = torch.mean(input, dim=-1, keepdim=True)
        variance = torch.var(input, dim=-1, unbiased=False, keepdim=True)
        output = (input - mean) / torch.sqrt(variance + eps)

        # ## step3: scale and shift (element-wise)
        if weight is not None:
            assert weight.shape == normalized_shape
            output = output * torch.flatten(weight)
        if bias is not None:
            assert bias.shape == normalized_shape
            output = output + torch.flatten(bias)

        # ## step4: 修改 output 张量的 shape
        output = output.view(input_shape)
        return output


class TestNativeLayerNorm(TestCase):
    """ NativeLayerNorm 不是用来使用的, 只是用来学习的 """

    def test_basic(self):
        layer_norm1 = NativeLayerNorm(normalized_shape=(3, 5), eps=1., elementwise_affine=True).eval()
        layer_norm2 = nn.LayerNorm(normalized_shape=[3, 5], eps=1., elementwise_affine=True).eval()

        with torch.no_grad():
            # 初始化一下, 不然有没有 "反标准化" 效果是一致的
            nn.init.xavier_normal_(layer_norm1.weight)
            nn.init.xavier_normal_(layer_norm1.bias)
            layer_norm1.weight.copy_(layer_norm2.weight)
            layer_norm1.bias.copy_(layer_norm2.bias)

        test_input = torch.randn(2, 4, 3, 5)
        result1 = layer_norm1(test_input)
        result2 = layer_norm2(test_input)

        self.assertTrue(
            is_same_tensor(result1, result2, eps=1e-6)
        )


class TestConditionalLayerNorm(TestCase):
    """ ConditionalLayerNorm 分为: 原本的用法 (emotion) 和论文中的用法 (token_pair) """

    def test_emotion(self):
        # 词嵌入的维度是 512, "心情" 嵌入的维度是 20
        word_embed_size, emotion_embed_size = 512, 20
        cln = ConditionalLayerNorm(
            input_size=word_embed_size, condition_size=emotion_embed_size, condition_hidden_size=256).eval()

        batch_size, num_words = 2, 10
        word_embed = torch.randn(batch_size, num_words, word_embed_size)
        emotion_embed = torch.randn(batch_size, emotion_embed_size)  # 一个句子一个 "心情"
        new_word_embed = cln(word_embed, condition=emotion_embed)

        self.assertEqual(new_word_embed.shape, word_embed.shape)

        # 由于初始化的原因, 一开始有没有 condition 效果应该是一致的
        layer_norm = nn.LayerNorm(normalized_shape=word_embed_size, elementwise_affine=True)
        new_word_embed2 = layer_norm(word_embed)
        self.assertTrue(is_same_tensor(new_word_embed, new_word_embed2, eps=1e-6))

    def test_word_pair(self):
        # 在 W2NER 中, condition 表示的是 "词语", 那么 condition_size = word_embed_size
        # 由于 condition 的 shape 是 [batch_size, num_words, word_embed_size],
        # 经过线性层和 unsqueeze 得到的 weight 和 bias 是 [batch_size, 1, num_words, word_embed_size]
        # 如果 input 的 shape 是 [batch_size, num_words, 1, word_embed_size]
        # 那么就可以得到 token-pair 级别 "反标准化" 的结果
        word_embed_size = 512
        cln = ConditionalLayerNorm(
            input_size=word_embed_size, condition_size=word_embed_size).eval()

        batch_size, num_words = 2, 10
        word_embed = torch.randn(batch_size, num_words, word_embed_size)

        grid = cln(torch.unsqueeze(word_embed, dim=2), condition=word_embed)
        self.assertEqual(grid.shape, (batch_size, num_words, num_words, word_embed_size))

        # 由于初始化的原因, 一开始有没有 condition 效果应该是一致的
        layer_norm = nn.LayerNorm(normalized_shape=word_embed_size, elementwise_affine=True)
        new_word_embed = layer_norm(word_embed)
        grid2 = torch.unsqueeze(new_word_embed, dim=2).expand(batch_size, num_words, num_words, word_embed_size)
        self.assertTrue(is_same_tensor(grid, grid2, eps=1e-6))
