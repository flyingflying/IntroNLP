# Author: lqxu

"""
测试 FAISS (Facebook AI Similarity Search)

安装: conda install -c pytorch faiss-cpu
GitHub: https://github.com/facebookresearch/faiss
paper: https://arxiv.org/pdf/1702.08734.pdf
"""

import torch
import faiss

from applications import SentenceEncoderV1


if __name__ == '__main__':

    # 对句子进行编码
    test_sentences = ["今天天气不错", "后天的天气怎么样呢", "这一周都下雨吗", "好耶！明天是晴天"]
    encoder = SentenceEncoderV1()
    sen_matrix = encoder(test_sentences)
    sen_matrix = torch.nn.functional.normalize(sen_matrix, p=2, dim=1)

    # 建立索引
    engine = faiss.IndexFlatIP(encoder.output_dims)
    engine.add(sen_matrix.numpy().astype("float32"))  # noqa

    query_matrix = encoder(["今天天气真好", "哎, 世界末日要到了"])
    query_matrix = torch.nn.functional.normalize(query_matrix, p=2, dim=1)
    print(engine.search(query_matrix.numpy().astype("float32"), 4))
    # print(engine.search(encoder(["今天天气真好"])[0], 1))
    # faiss.write_index(engine, "./test_faiss.bin")
