# Author: lqxu

from typing import *

import torch
from torch import Tensor

from itertools import groupby, combinations


class GPLinkerEEScheme:

    def __init__(self, argument_labels: List[Tuple[str, str]], ensure_trigger: bool = True):
        self.argument_labels = argument_labels
        self.num_labels = len(argument_labels)
        self.ensure_trigger = ensure_trigger  # 解码时, 是否要保证触发词的存在

    def encode(self, sample: Dict[str, Any]):

        input_ids = sample["input_ids"]
        num_tokens = len(input_ids)

        arguments_tensor = torch.zeros(self.num_labels, num_tokens, num_tokens)
        heads_tensor = torch.zeros(num_tokens, num_tokens)
        tails_tensor = torch.zeros(num_tokens, num_tokens)

        for event in sample["events"]:
            for event_type, argument_role, head, tail in event:
                label_id = self.argument_labels.index((event_type, argument_role))
                arguments_tensor[label_id, head, tail] = 1

            """
            heads 和 tails 矩阵的构建和关系抽取是相似的。
            在关系抽取中, 我们一般会定义 dim=0 是 subject 索引, dim=1 是 object 索引, 始终是 subject 和 object 相连。
            在事件抽取中, 是一组论元之间的关系, dim=0 和 dim=1 没有明确的定义, 只要保证一组论元中两两相连即可。为了保证唯一性, 做出以下限制:
                两个论元相连时, dim=0 维度上的值较小, dim=1 维度上的索引值较大。
            举例来说, 现在有两个论元, 分别是 (1, 2) 和 (3, 4), 对于 heads 来说, 其有两种连线方式:
                a) 1 和 3 连线
                b) 3 和 1 连线
            为了保证唯一性, 我们现在只取 a) 方式, 将 b) 方式给省略掉。
            因此, 对于 heads/tails 矩阵来说, 我们只需要上三角, 下三角是不需要的。  
            """
            for (_, _, head1, tail1), (_, _, head2, tail2) in combinations(event, 2):
                heads_tensor[min(head1, head2), max(head1, head2)] = 1
                tails_tensor[min(tail1, tail2), max(tail1, tail2)] = 1

        return arguments_tensor, heads_tensor, tails_tensor

    def decode(self, arguments_tensor: Tensor, heads_tensor: Tensor, tails_tensor: Tensor):

        # step1: 解码 arguments
        arguments = set()  # 论元的形式: (event_type, argument_role, head_idx, tail_idx)
        for label_id, head, tail in zip(*torch.where(arguments_tensor == 1)):
            label_id, head, tail = label_id.item(), head.item(), tail.item()
            if head > tail:
                continue
            arguments.add(self.argument_labels[label_id] + (head, tail))

        # step2: 构建图
        links = set()  # 链接的形式: (head_idx_1, tail_idx_1, head_idx_2, tail_idx_2)
        for (_, _, head1, tail1), (_, _, head2, tail2) in combinations(arguments, 2):
            if heads_tensor[min(head1, head2), max(head1, head2)] == 0:
                continue
            if tails_tensor[min(tail1, tail2), max(tail1, tail2)] == 0:
                continue
            # 这里是 双向链接!
            links.add((head1, tail1, head2, tail2))
            links.add((head2, tail2, head1, tail1))

        # step3: 完全子图搜索
        events = []
        """ 
        注意这里的 arguments 首先根据 event_type 进行聚类, 即每一个 event_type 内找完全子图 
        
        注意, python 的 itertools 中的 groupby 不是 **分组**, 而是 **相邻分组**, 因此一定要先排序, 坑啊 !!!
        """
        for _, sub_arguments in groupby(sorted(arguments), key=lambda a: a[0]):
            sub_arguments = tuple(sorted(sub_arguments))
            events.extend(self.clique_search(sub_arguments, links))

        # step4: 检测出发词的存在
        if self.ensure_trigger:
            events = [event for event in events if any(argument[1] == "触发词" for argument in event)]

        return events

    def clique_search(self, all_nodes, links):

        """
        完全图(complete graph) / 团(clique) 搜索 \n
        每一个 argument 就是一个节点, 每一个 event 就是一个 complete graph / clique \n
        :param all_nodes: 同一事件类型的所有节点, 也就是一个 argument 类型的集合
        :param links: 所有的边集合
        :return: 事件/完全图/团 的集合, 每一个事件由若干论元组成, 类型可以写成: List[List[Tuple[str, str, int, int]]]
        """

        sub_graphs = set()

        for node1, node2 in combinations(all_nodes, 2):  # 枚举所有的节点对
            if node1[2:] + node2[2:] not in links:  # 如果存在不相邻的节点对
                sub_graphs.add(self.find_neighbors(node1, all_nodes, links))
                sub_graphs.add(self.find_neighbors(node2, all_nodes, links))
        # 由于是枚举所有的节点对, 因此可能会找到多个相同的子图, 那么就需要用 set 进行子图的去重

        if len(sub_graphs) == 0:
            return {all_nodes, }

        results = set()
        for sub_graph in sub_graphs:
            # TODO: 这里使用了递归, 理论上不会出问题, 以后改成循环或者进行最大递归数限制
            results.update(self.clique_search(sub_graph, links))
        return results

    @staticmethod
    def find_neighbors(main_node, all_nodes, links):
        """
        找到所有和 main_node 相连的节点 \n
        在这里, 每一个节点都是一个 argument, 数据形式为: (event_type, argument_role, head_idx, tail_idx) \n
        :param main_node: 主节点 (一个 argument)
        :param all_nodes: 所有的节点 (argument 列表)
        :param links: 所有的边集合, 数据形式为: (head_idx_1, tail_idx_1, head_idx_2, tail_idx_2)
        :return: 和 main_node 相邻的所有节点 (argument 列表)
        """

        neighbors = {node for node in all_nodes if node[2:] + main_node[2:] in links}
        neighbors.add(main_node)  # 自己一定要在 sub_graph 中
        # 用 sorted 排序和 tuple 封装是为了保证能添加入 set 中, 进行 sub_graph 去重
        return tuple(sorted(neighbors))


if __name__ == '__main__':

    from data_modules import argument_labels as argument_labels_

    # test_sample_ = {
    #     "input_ids": [  # text 编码后的 ID 值
    #         101, 679, 788, 788, 3221, 704, 1744, 8233, 821, 689, 1762, 6161, 1447, 8024, 711, 862, 8195, 2487, 4638,
    #         4508, 7755, 3152, 738, 1355, 4495, 749, 1059, 4413, 6161, 1447, 102,
    #     ],
    #     "events": [  # 事件列表
    #         [  # 一个事件由若干论元组成 (触发词也当作一个论元)
    #             ["组织关系-裁员", "裁员方", 5, 9],  # 一个论元: (event_type, argument_role, start_idx/head, end_idx/tail)
    #             ["组织关系-裁员", "触发词", 11, 12]
    #         ],
    #         [
    #             ["组织关系-裁员", "裁员方", 16, 21],
    #             ["组织关系-裁员", "触发词", 28, 29]
    #         ]
    #     ]
    # }
    #
    # test_sample_ = {
    #     'input_ids': [
    #         101, 2945, 4338, 7987, 7390, 7339, 6381, 5442, 13269, 8613, 8983, 10861, 10039, 2845, 6887, 8024,
    #         3867, 2622, 782, 1894, 6851, 7463, 8024, 4338, 7987, 2347, 2199, 1184, 7226, 5855, 1217, 2349, 118,
    #         4906, 5287, 4294, 6161, 2957, 511, 3634, 1184, 800, 680, 4338, 7987, 5041, 678, 749, 671, 819, 9577,
    #         8963, 9407, 8108, 1394, 1398, 511, 1762, 6158, 6161, 2957, 1400, 8024, 4906, 5287, 4294, 678, 6612,
    #         2108, 1920, 3519, 4372, 2199, 1184, 2518, 4338, 7987, 4638, 1355, 2245, 5468, 4673, 4413, 7339, 3126,
    #         1213, 511, 102
    #     ],
    #     'events': [
    #         [
    #             ['组织关系-裁员', '裁员方', 23, 24],
    #             ['组织关系-裁员', '触发词', 36, 37]
    #         ],
    #         [
    #             ['组织关系-加盟', '加盟者', 27, 35],
    #             ['组织关系-加盟', '所加盟组织', 43, 44],
    #             ['组织关系-加盟', '触发词', 45, 46]
    #         ],
    #         [
    #             ['组织关系-裁员', '裁员方', 23, 24],
    #             ['组织关系-裁员', '触发词', 59, 60]
    #         ]
    #     ]
    # }

    test_sample_ = {
        'input_ids': [
            101, 6818, 1126, 2399, 4638, 2108, 1400, 6612, 1377, 6458, 6414, 4495, 749, 679, 2208, 4638, 4868,
            6839, 8024, 1071, 704, 3300, 763, 4413, 3215, 2802, 1139, 749, 1380, 6408, 5663, 4638, 3144, 2945, 8024,
            3300, 763, 4413, 7339, 1156, 1158, 6863, 749, 1325, 1380, 5279, 2497, 8024, 1762, 6821, 702, 6382, 4955,
            3144, 2945, 4638, 2399, 807, 8024, 4507, 2342, 3215, 2802, 1139, 1290, 714, 4638, 3144, 2945, 738, 679,
            6639, 711, 1936, 8024, 852, 1762, 8121, 2399, 2600, 1104, 6612, 6929, 702, 5307, 1073, 4638, 5143, 1154,
            6612, 7027, 8024, 2382, 6226, 6612, 2802, 1139, 8454, 5526, 130, 6566, 4638, 1325, 1380, 5018, 671, 2773,
            5327, 4638, 1235, 1894, 1316, 1762, 1920, 3683, 1146, 124, 118, 122, 7566, 1044, 2658, 1105, 678, 6158,
            6285, 1990, 102
        ],
        # (92, 110) 的位置实体类型既是 胜者, 又是 败者, 没有办法避免, 一定会多出来两个事件 !!!
        'events': [
            [
                ['竞赛行为-胜负', '胜者', 92, 110],
                ['竞赛行为-胜负', '赛事名称', 92, 94],
                ['竞赛行为-胜负', '触发词', 98, 98]
            ],
            [
                ['竞赛行为-胜负', '败者', 92, 110],
                ['竞赛行为-胜负', '赛事名称', 92, 94],
                ['竞赛行为-胜负', '触发词', 100, 100]
            ],
            [
                ['竞赛行为-胜负', '败者', 92, 110],
                ['竞赛行为-胜负', '赛事名称', 77, 89]
            ]
        ]
    }

    scheme_ = GPLinkerEEScheme(argument_labels_)
    arguments_tensor_, heads_tensor_, tails_tensor_ = scheme_.encode(test_sample_)

    gold_events = [tuple(sorted(tuple(argument) for argument in gold_event)) for gold_event in test_sample_["events"]]

    pred_events = scheme_.decode(arguments_tensor_, heads_tensor_, tails_tensor_)
    print("---" * 10)
    for event_ in pred_events:
        print(event_)
        print(event_ in gold_events)
