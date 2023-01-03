# Author: lqxu

""" 苏剑林大佬提出的 Global Pointer 模块 !!! 代码是早期在 jupyter 上写的, 风格可以能会有一些不一样, 请见谅 """

import math

import torch
from torch import nn, Tensor

from transformers.models.bert import BertConfig

__all__ = ["RotaryPositionEmbedding", "GlobalPointer", "EfficientGlobalPointer"]


class RotaryPositionEmbedding(nn.Module):
    """
    旋转式位置编码: https://kexue.fm/archives/8265 \n
    简单来说就是: 计算 attention 时, query @ key 的结果中包含位置信息
    """

    cos_part: Tensor  # [max_length, head_size]
    sin_part: Tensor  # [max_length, head_size]

    def __init__(self, config: BertConfig, head_size: int = 64):
        # ## 参数检查
        assert head_size % 2 == 0, "head size must be an even number!!!"
        super().__init__()
        max_seq_length = config.max_position_embeddings

        sinusoid_embeddings = self.get_sinusoid_encoding_table(max_seq_length=max_seq_length, embedding_size=head_size)
        # numpy 中的 repeat 和 torch 中的 repeat_interleave 是一致的
        cos_part: Tensor = sinusoid_embeddings[:, 1::2].repeat_interleave(2, dim=-1)  # [max_length, head_size]
        sin_part: Tensor = sinusoid_embeddings[:, 0::2].repeat_interleave(2, dim=-1)  # [max_length, head_size]

        # 将 sin_part 偶数 (0-based) 索引位置全部加上负号
        tmp = sin_part.view(max_seq_length, -1, 2)
        tmp[:, :, 0] = -tmp[:, :, 0]
        # sin_part = tmp.view(max_seq_length, -1)  # 不需要 view 回去, 因为 sin_part 中的值已经修改了

        self.register_buffer("cos_part", cos_part, persistent=True)
        self.register_buffer("sin_part", sin_part, persistent=True)

    def forward(self, qk_tensor: Tensor):
        # qk_tensor: 计算 attention 时的 query 或者 key 矩阵, 维度是 [batch_size, num_heads, seq_len, head_size]
        prefix_shape = qk_tensor.shape[:-1]
        seq_len = qk_tensor.shape[-2]
        # stack 后的结果是: [batch_size, num_heads, seq_len, head_size/2, 2], view_as 将最后两个维度组合在一起
        # qk_tensor_2 = torch.stack([-qk_tensor[..., 1::2], qk_tensor[..., 0::2]], dim=-1).view_as(qk_tensor)
        # torch 中不支持 negative step, 那就只能用 flip 函数来替代
        qk_tensor_2 = qk_tensor.view(*prefix_shape, -1, 2).flip(dims=[-1]).view(*prefix_shape, -1)
        return qk_tensor * self.cos_part[:seq_len, :] + qk_tensor_2 * self.sin_part[:seq_len, :]

    @staticmethod
    def get_sinusoid_encoding_table(max_seq_length: int = 512, embedding_size: int = 768) -> Tensor:
        """这个函数是初版 transformer 中的位置编码方式 (sinusoid 正弦波)"""
        # ## 初始化位置索引值
        position = torch.arange(start=0, end=max_seq_length).float().unsqueeze(1)  # [max_length, 1]
        # ## 求 theta 的值: 先取对数, 再用指数还原
        theta = torch.exp(  # [embedding_size / 2]
            -torch.arange(start=0, end=embedding_size, step=2).float() / embedding_size * math.log(10000)
        )
        # ## 构建嵌入表格
        embeddings_table = torch.zeros(max_seq_length, embedding_size).float()
        embeddings_table[:, 0::2] = torch.sin(position * theta)  # 偶数位
        embeddings_table[:, 1::2] = torch.cos(position * theta)  # 奇数位
        return embeddings_table


class GlobalPointer(nn.Module):
    """
    全局指针模块: https://kexue.fm/archives/8373 \n
    按照 BertSelfAttention 的架构重写
    """
    def __init__(self, config: BertConfig, num_heads: int, head_size: int = 64, use_rope: bool = True):
        """
        :param num_heads: 按照苏剑林的设计, 一个 head 对应一个标签
        :param head_size: 一个 head 的维度 (默认是 64, 和 BERT 的设计保持一致, 即 768 / 12)
        :param use_rope: 是否使用 RoPE 进行位置编码
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.norm_value = self.head_size ** 0.5
        self.all_head_size = self.num_heads * self.head_size
        self.query = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size, bias=True)
        self.key = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size, bias=True)
        self.position_embedding = RotaryPositionEmbedding(config) if use_rope else None

    def forward(self, hidden_states: Tensor) -> Tensor:
        # ## step1: 计算 query_layer 和 key_layer
        # 两者的 shape 都是: [batch_size, num_heads, seq_len, head_size]
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))

        # ## step2: 添加位置编码信息
        if self.position_embedding is not None:
            query_layer = self.position_embedding(query_layer)
            key_layer = self.position_embedding(key_layer)

        # ## step3: 计算 logits 值
        # 其 shape 为: [batch_size, num_heads, seq_len_query, seq_len_key]
        logits = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 这里和 BertSelfAttention 中的代码保持一致, 如果按照 tianshan1994 的代码,
        # 那么这里的 query_layer 和 key_layer 的维度应该都是 [batch_size, seq_len, num_heads, head_size],
        # 此时可以用爱因斯坦求和法, 那么代码应该是这个的:
        # attention_scores = torch.einsum('bmhd,bnhd->bhmn', query_layer, key_layer)

        return logits / self.norm_value

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]


class EfficientGlobalPointer(nn.Module):
    """
    高效全局指针模块: https://kexue.fm/archives/8877 \n
    没有按照 BertSelfAttention 的架构来写, 按照我自己的理解来写的
    """

    def __init__(self, config: BertConfig, num_tags: int, head_size: int = 64, use_rope: bool = True):
        """
        :param num_tags: 按照苏剑林的设计, 一个 head 对应一个标签
        :param head_size: 一个 head 的维度 (默认是 64, 和 BERT 的设计保持一致, 即 768 / 12)
        :param use_rope: 是否使用 RoPE 进行位置编码
        """
        super().__init__()

        self.num_tags = num_tags
        self.head_size = head_size
        self.hidden_size = config.hidden_size
        self.norm_value = self.head_size ** 0.5

        self.start_transform = nn.Linear(in_features=self.hidden_size, out_features=head_size, bias=True)
        self.end_transform = nn.Linear(in_features=self.hidden_size, out_features=head_size, bias=True)
        self.position_embedding = RotaryPositionEmbedding(config) if use_rope else None
        # 这里使用了技巧, 按照原博客, 应该是下面这种写法, 具体见 code_test 方法
        # self.cls_transform = nn.Linear(in_features=head_size * 4, out_features=num_tags)
        self.cls_transform = nn.Linear(in_features=head_size * 2, out_features=num_tags * 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = hidden_states.shape
        start_tensor = self.start_transform(hidden_states)  # [batch_size, seq_len, head_size]
        end_tensor = self.end_transform(hidden_states)      # [batch_size, seq_len, head_size]
        if self.position_embedding is not None:
            start_tensor = self.position_embedding(start_tensor)  # [batch_size, seq_len, head_size]
            end_tensor = self.position_embedding(end_tensor)      # [batch_size, seq_len, head_size]

        # ## 计算识别的 logits 值
        # recognition_logits: [batch_size, seq_len_start, seq_len_end]
        # recognition_logits = torch.matmul(start_tensor, end_tensor.transpose(-1, -2)) / self.norm_value
        recognition_logits = torch.einsum("bsh,beh->bse", start_tensor, end_tensor) / self.norm_value

        # ## 计算分类的 logits 值
        # 用 start_tensor 和 end_tensor 拼接的方式替代 hidden_states, 以减少参数量
        # cls_input = hidden_states
        cls_input = torch.cat([start_tensor, end_tensor], dim=-1)  # [batch_size, seq_len, head_size * 2]
        # 这里除以 2 仅仅是为了防止数字过大, 和上面 recognition_logits 除以 norm_value 是同样的目的
        cls_logits = self.cls_transform(cls_input) / 2             # [batch_size, seq_len, num_tags * 2]
        # cls_logits' shape: [batch_size, num_tags, seq_len, 2]
        cls_logits = cls_logits.view(batch_size, seq_len, self.num_tags, 2).permute(0, 2, 1, 3)
        # cls_logits' shape: # [batch_size, num_tags, seq_len, seq_len]
        cls_logits = cls_logits[:, :, :, :1] + cls_logits[:, :, :, 1:].transpose(-1, -2)

        # ## 将两个 logits 值加在一起, 作为最终的 logits 值
        logits = recognition_logits.unsqueeze(1) + cls_logits  # [batch_size, num_tags, seq_len, seq_len]
        logits = logits.contiguous()  # 居然不连续了 ...
        return logits

    @staticmethod
    def code_test():
        # ## 工具函数
        def is_same_tensor(tensor1, tensor2, theta=1e-6):
            return torch.all(torch.abs(tensor1 - tensor2) < theta)

        def create_tensor(*size):
            if len(size) == 0:
                return torch.tensor(0)
            from functools import reduce
            from operator import mul
            return torch.arange(reduce(mul, size)).reshape(*size)

        # ## 要点: 拼接向量的运算是可以拆分的
        input1 = torch.randn(5)
        input2 = torch.randn(5)
        weight = torch.rand(8, 10)
        print(is_same_tensor(
            weight @ torch.cat([input1, input2], dim=-1),     # [8, ] 15.1 us
            weight[:, :5] @ input1 + weight[:, 5:] @ input2,  # [8, ] 32.3 us
        ))

        # ## 扩展到矩阵的情况
        input1 = torch.randn(4, 5)  # [batch_size, in_features]
        input2 = torch.randn(4, 5)  # [batch_size, in_features]
        weight = torch.randn(8, 10)  # [out_features, in_features]
        print(is_same_tensor(
            torch.cat([input1, input2], dim=-1) @ weight.T,      # [batch_size, out_features]=[4, 8]
            input1 @ weight[:, :5].T + input2 @ weight[:, 5:].T  # [batch_size, out_features]=[4, 8]
        ))

        # ## 原版代码
        # 设 batch_size = 3, seq_len = 4, head_size = 5, num_tags = 2, 那么
        batch_size, seq_len, head_size, num_tags = 3, 4, 5, 2
        input_ = create_tensor(batch_size, seq_len, head_size * 2)  # [3, 4, 10]
        weight = create_tensor(num_tags, head_size * 4)  # [2, 20]
        # forward 代码 (约 158 us)
        cls_input = torch.concat(
            [
                input_.repeat_interleave(seq_len, dim=1),  # [batch_size, seq_len * seq_len, head_size * 2]=[3, 16, 10]
                input_.repeat(1, seq_len, 1)  # [batch_size, seq_len * seq_len, head_size * 2]=[3, 16, 10]
            ], dim=-1
        ).reshape(batch_size, seq_len, seq_len, head_size * 4)  # [3, 4, 4, 20]
        logits1 = cls_input @ weight.T  # [batch_size, seq_len, seq_len, num_tags]=[3, 4, 4, 2]
        logits1 = logits1.permute(0, 3, 1, 2)  # [batch_size, seq_len, seq_len, num_tags]=[3, 2, 4, 4]

        # ## 新版代码
        weight_new = weight.view(num_tags * 2, head_size * 2)  # [4, 10]
        # forward 代码 (约 64 us)
        logits2 = input_ @ weight_new.T  # [batch_size, seq_len, num_tags * 2]=[3, 4, 4]
        logits2 = logits2.view(batch_size, seq_len, num_tags, 2).permute(0, 2, 1, 3)
        logits2 = logits2[:, :, :, :1] + logits2[:, :, :, 1:].transpose(-1, -2)

        print(is_same_tensor(logits1, logits2))


if __name__ == '__main__':
    EfficientGlobalPointer.code_test()
