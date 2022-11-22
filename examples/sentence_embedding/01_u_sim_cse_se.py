# Author: lqxu

"""
改自 sentence-transformers 2.2.2 库中的样例, 测试 unsupervised SimCSE + OCNLI + LCQMC 的效果, 确保效果确实不好

Reference:
    https://github.com/UKPLab/sentence-transformers/blob/v2.2.2/examples/unsupervised_learning/SimCSE/train_stsb_simcse.py

主要的库依赖: sentence-transformers==2.2.2
"""

import _prepare  # noqa

import os
import math
import logging

from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from core import utils

logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])


if __name__ == '__main__':

    # 训练参数
    model_name = 'hfl/chinese-roberta-wwm-ext'
    train_batch_size = 128
    num_epochs = 1
    max_seq_length = 32

    # 模型保存的路径
    model_save_path = os.path.join(utils.ROOT_DIR, "outputs/u_sim_cse_se")

    # 初始化模型
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("加载训练数据")
    train_samples = []
    with open(os.path.join(utils.DATA_DIR, "sentence_embeddings/LCQMC/train.txt"), "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            sen1, sen2, _ = line.split("\t")
            if len(sen1) >= 10: train_samples.append(InputExample(texts=[sen1, sen1]))  # noqa: E701
            if len(sen2) >= 10: train_samples.append(InputExample(texts=[sen2, sen2]))  # noqa: E701

    logging.info("加载验证和测试数据")
    dev_samples, test_samples = [], []
    with open(os.path.join(utils.DATA_DIR, "sentence_embeddings/LCQMC/dev.txt"), "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            sen1, sen2, label = line.split("\t")
            dev_samples.append(InputExample(texts=[sen1, sen2], label=float(label)))

    with open(os.path.join(utils.DATA_DIR, "sentence_embeddings/LCQMC/test.txt"), "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            sen1, sen2, label = line.split("\t")
            test_samples.append(InputExample(texts=[sen1, sen2], label=float(label)))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, batch_size=train_batch_size, name='LCQMC-dev')
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, batch_size=train_batch_size, name='LCQMC-test')

    # We train our model using the MultipleNegativesRankingLoss
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)  # noqa
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% 的数据用于 warmup
    evaluation_steps = int(len(train_dataloader) * 0.1)  # 每 10% 步数进行一次验证
    logging.info(f"训练的句子数是 {len(train_samples)}")
    logging.info(f"warmup 步数 {warmup_steps}")
    logging.info("训练之前的性能")

    dev_evaluator(model)
    test_evaluator(model, output_path=model_save_path)

    # 开始训练
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              optimizer_params={'lr': 5e-5},
              use_amp=True
              )

    model = SentenceTransformer(model_save_path)
    test_evaluator(model, output_path=model_save_path)
