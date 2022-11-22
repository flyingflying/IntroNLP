# Author: lqxu

from unittest import TestCase

import torch

from core import utils
from core.utils import vector_pair


class PairwiseTest(TestCase):
    def test_pairwise_cosine_similarity(self):
        input1 = torch.randn(15, 20)
        input2 = torch.randn(5, 20)

        result1 = vector_pair.pairwise_cosine_similarity(input1, input2)
        result2 = vector_pair.pairwise_dot_product(
            torch.nn.functional.normalize(input1, p=2, dim=-1, eps=1e-12),
            torch.nn.functional.normalize(input2, p=2, dim=-1, eps=1e-12)
        )
        self.assertTrue(utils.is_same_tensor(result1, result2))

    def test_pairwise_distance(self):
        # 测试曼哈顿距离
        num_samples1, num_samples2, num_features = 3, 4, 5
        input1 = torch.randn(num_samples1, num_features)
        input2 = torch.randn(num_samples2, num_features)

        result = vector_pair.pairwise_distance(input1, input2, p=1.)

        for idx1 in range(num_samples1):
            vector1 = input1[idx1]
            for idx2 in range(num_samples2):
                vector2 = input2[idx2]
                distance = torch.sum(torch.abs(vector1 - vector2))
                self.assertAlmostEqual(
                    result[idx1, idx2].item(), distance.item(), delta=1e-5)

        pass
