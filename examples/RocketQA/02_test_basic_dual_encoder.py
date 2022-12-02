# Author: lqxu

import _prepare  # noqa

import os
import time
import logging
from typing import *

import torch
import faiss  # noqa
import numpy as np
from tqdm import tqdm  # noqa

from core.utils import ROOT_DIR
from core.models import SentenceBertModel
from core.utils import get_default_tokenizer, BigFileReader, read_json, to_json

logging.basicConfig(level=logging.INFO)  # 必须要这样才能在 console 中显示 info 文本

raw_data_dir: str = os.path.join(ROOT_DIR, "./datasets/du_reader_retrieval/")
phase2_root_dir = os.path.join(ROOT_DIR, "./examples/RocketQA/02/")
card_pre_num = 2024167  # 四张卡的数据量是相同的, 可以通过 `cat fil_name | wc -l` 获得


@torch.no_grad()
def main_build_engine(batch_size: int = 1024, device: str = "cuda:0"):

    # 构建分词器
    tokenizer = get_default_tokenizer()
    tokenizer_kwargs = {
        "padding": "max_length", "truncation": True, "max_length": 384,
        "return_attention_mask": False, "return_token_type_ids": False, "return_tensors": "pt"}

    # 加载模型
    model_path = os.path.join(phase2_root_dir, f"model/passage_encoder")
    model: SentenceBertModel = SentenceBertModel.from_pretrained(model_path).eval().to(device)

    for card in range(4):
        buffer = []
        engine = faiss.IndexFlatIP(768)
        passage_collection_path = os.path.join(raw_data_dir, f"./passage-collection/part-0{card}")

        with open(passage_collection_path, "r", encoding="utf-8") as reader:
            pbar = tqdm(total=card_pre_num)  # 四个 card 中的数据量时相同的
            for line in reader:
                buffer.append(line[3:-3])
                if len(buffer) == batch_size:
                    # 分词
                    p_input_ids: torch.Tensor = tokenizer(buffer, **tokenizer_kwargs)["input_ids"].to(device)
                    # 计算句向量
                    query_vector = model(p_input_ids)[1].detach().cpu().numpy()
                    # 添加入搜索引擎中
                    engine.add(query_vector)  # noqa
                    buffer.clear()
                pbar.update(1)
            pbar.close()

        if buffer:
            p_input_ids: torch.Tensor = tokenizer(buffer, **tokenizer_kwargs)["input_ids"].to(device)
            engine.add(model(p_input_ids)[1].detach().cpu().numpy())  # noqa

        # 保存引擎结果到文件中
        faiss.write_index(
            engine,
            os.path.join(phase2_root_dir, "engine", f"engine_part_{card}.bin")
        )


def show_passage(passage_indices: List[int]):
    """ 展示训练完成的文本 """

    logger = logging.getLogger("test search logger")
    big_readers = []

    logger.info("开始读取大文件")
    for card in range(4):
        passage_collection_path = os.path.join(raw_data_dir, f"./passage-collection/part-0{card}")
        cache_path = f"{passage_collection_path}.cache"
        start = time.time()
        big_reader = BigFileReader(passage_collection_path, cache_path)
        end = time.time()
        # 第一次调用的话, 需要建立 offset 索引, 一个文件会有 5.5 分钟的耗时
        logger.info(f"读取完成第 {card} 文件, 共耗时 {round(end - start, 2)} 秒 !!!")
        big_readers.append(big_reader)

    for passage_index in passage_indices:
        card = passage_index // card_pre_num
        line_no = passage_index % card_pre_num
        print(big_readers[card].read_specified_line(line_no)[3:-3])


@torch.no_grad()
def test_search(device: str = "cuda:0", top_k: int = 50):

    logger = logging.getLogger("test search logger")
    test_query = "哪个网站有湖南卫视直播"

    # 构建分词器
    tokenizer = get_default_tokenizer()
    tokenizer_kwargs = {
        "padding": "max_length", "truncation": True, "max_length": 32,
        "return_attention_mask": False, "return_token_type_ids": False, "return_tensors": "pt"}

    # 加载模型
    model_path = os.path.join(phase2_root_dir, f"model/query_encoder")
    model: SentenceBertModel = SentenceBertModel.from_pretrained(model_path).eval().to(device)

    # 加载搜索引擎
    logger.info("开始加载搜索索引 ... ")  # 4 个引擎大约占 24 GB 的内存, 加载一个大约要 6.5 - 7 秒的时间
    engines = []
    for card in range(4):
        start_time = time.time()
        engine = faiss.read_index(os.path.join(phase2_root_dir, "engine", f"engine_part_{card}.bin"))
        end_time = time.time()
        logger.info(f"成功加载第 {card} 个搜索引擎, 耗时 {round(end_time-start_time, 2)} 秒 !!!")
        engines.append(engine)

    # 对 query 进行编码
    q_input_ids = tokenizer(test_query, **tokenizer_kwargs)["input_ids"].to(device)
    q_vector = model(q_input_ids)[1].detach().cpu().numpy()

    # 检索 top-k 的内容
    scores, indices = [], []
    for card, engine in enumerate(engines):
        part_scores, part_indices = engine.search(q_vector, top_k)
        scores.append(part_scores[0])
        indices.append(part_indices[0] + card_pre_num * card)
    scores = np.concatenate(scores, axis=0)
    indices = np.concatenate(indices, axis=0)
    passage_indices = indices[np.argsort(scores)[::-1]][:top_k].tolist()

    show_passage(passage_indices)


@torch.no_grad()
def main_search(device: str = "cuda:0", top_k: int = 100):

    logger = logging.getLogger("main search logger")

    # 加载 passage2id 映射表
    logger.info("开始加载文章映射表")
    file_path = os.path.join(raw_data_dir, "passage-collection/passage2id.map.json")
    passage2id = read_json(file_path)
    passage2id: Dict[int, str] = {int(key): value for key, value in passage2id.items()}  # 将 key 值从字串转化为整型
    logger.info("成功加载文章映射表")

    # 加载分词器
    logger.info("开始加载分词器")
    tokenizer = get_default_tokenizer()
    tokenizer_kwargs = {
        "padding": "max_length", "truncation": True, "max_length": 32,
        "return_attention_mask": False, "return_token_type_ids": False, "return_tensors": "pt"}
    logger.info("分词器加载完成")

    # 加载模型
    logger.info("开始加载模型")
    model_path = os.path.join(phase2_root_dir, f"model/query_encoder")
    model: SentenceBertModel = SentenceBertModel.from_pretrained(model_path).eval().to(device)
    logger.info("模型加载完成")

    # 加载搜索引擎
    logger.info("开始加载搜索索引 ... ")  # 4 个引擎大约占 24 GB 的内存, 加载一个大约要 6.5 - 7 秒的时间
    engines = []
    for card in range(4):
        engine = faiss.read_index(os.path.join(phase2_root_dir, "engine", f"engine_part_{card}.bin"))
        engines.append(engine)
        logger.info(f"成功加载第 {card} 个搜索引擎")

    # 开始预测
    results = {}
    dev_samples: Dict[str, str] = read_json(os.path.join(raw_data_dir, "dev/q2qid.dev.json"))
    for query_text, query_id in tqdm(dev_samples.items()):
        # 对 query 进行编码
        q_input_ids = tokenizer(query_text, **tokenizer_kwargs)["input_ids"].to(device)
        q_vector = model(q_input_ids)[1].detach().cpu().numpy()

        # 检索 top-k 的结果
        scores, indices = [], []
        for card, engine in enumerate(engines):
            part_scores, part_indices = engine.search(q_vector, top_k)
            scores.append(part_scores[0])
            indices.append(part_indices[0] + card_pre_num * card)
        scores = np.concatenate(scores, axis=0)
        indices = np.concatenate(indices, axis=0)
        passage_indices = indices[np.argsort(scores)[::-1]][:top_k].tolist()

        # 记录结果
        passage_ids = [passage2id[passage_index] for passage_index in passage_indices]
        results[query_id] = passage_ids

    to_json(os.path.join(phase2_root_dir, "dev_results.json"), results, readable=True)


if __name__ == '__main__':
    # main_build_engine()
    # test_search()
    main_search()
