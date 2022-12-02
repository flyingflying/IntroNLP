# Author: lqxu

"""
改自 sentence-transformers 2.2.2 库中的样例, 测试 SBERT + OCNLI + CNSD-STS-B 的效果, 确保效果确实不好

Reference:
    https://github.com/UKPLab/sentence-transformers/blob/v2.2.2/examples/training/nli/training_nli.py

主要的库依赖: sentence-transformers==2.2.2
"""

import _prepare  # noqa

import os
import json
import math
import logging

from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample

from core import utils

logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])


if __name__ == '__main__':

    # hyper params
    num_epochs = 1
    model_name = 'hfl/chinese-roberta-wwm-ext'
    train_batch_size = 16
    model_save_path = os.path.join(utils.ROOT_DIR, "./examples/sentence_embedding/outputs/sbert_nli_se")

    # model
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 读取数据
    logging.info("读取 OCNLI 数据集")
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    train_samples = []
    with open(os.path.join(utils.DATA_DIR, "sentence_embeddings/OCNLI/train.50k.jsonl"), "r") as reader:
        for line in reader:
            sample = json.loads(line)
            label_id = label2int.get(sample["label"], -100)
            train_samples.append(
                InputExample(texts=[sample["sentence1"], sample["sentence2"]], label=label_id)
            )

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)  # noqa
    train_loss = losses.SoftmaxLoss(
        model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))

    logging.info("读取 CNSD-STS-B 数据集")
    dev_samples = []
    with open(os.path.join(utils.DATA_DIR, "sentence_embeddings/STS-B/cnsd-sts-dev.txt"), "r") as reader:
        for line in reader:
            _, sen1, sen2, label = line.rstrip().split("||")
            score = float(label) / 5.0
            dev_samples.append(
                InputExample(texts=[sen1, sen2], label=score)
            )

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, batch_size=train_batch_size, name='sts-dev')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=100,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )

    test_samples = []
    with open(os.path.join(utils.DATA_DIR, "sentence_embeddings/STS-B/cnsd-sts-test.txt"), "r") as reader:
        for line in reader:
            _, sen1, sen2, label = line.rstrip().split("||")
            score = float(label) / 5.0
            test_samples.append(
                InputExample(texts=[sen1, sen2], label=score)
            )

    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, batch_size=train_batch_size, name='sts-test')
    test_evaluator(model, output_path=model_save_path)
