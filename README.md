# IntroNLP

个人的 NLP 实验算法库 和 笔记库。欢迎所有对 NLP 算法有兴趣的小伙伴。如果发现代码有任何的 bug, 欢迎提 issue !!! 我一定会认真看的。

目前统一使用 BERT 模型对词语和句子进行编码; 主要的开发工具是 HuggingFace Transformers 和 PyTorch-Lightning; 暂时只考虑中文。未来会有扩展的。

### 实验环境的搭建

目前本项目的 Python 版本是 3.8, 以后只会测试更高的 Python 版本, 不会测试低的版本。

```shell
pip install -r requirements.txt 
```

建议将 HuggingFace Transformers 的缓存路径写进 `bashrc` 或者 `zshrc` 中 (Windows 电脑加入环境变量中, 并开启开发者模式)。
这样所有预训练模型都会缓存在指定的目录中, 非常方便。更多请参考: [cache setup](https://huggingface.co/docs/transformers/v4.24.0/en/installation#cache-setup) 和 [cache management](https://huggingface.co/docs/datasets/cache) 。

```shell
export HUGGINGFACE_HUB_CACHE=""
export TRANSFORMERS_CACHE=""
export HF_DATASETS_CACHE=""
```

### 项目的目录结构

+ `core` 文件夹包含所有的模型类以及辅助训练的工具
+ `datasets` 文件夹包含所有的数据集说明和下载地址
+ `docs` 文件夹包含所有的相关笔记
+ `examples` 文件夹包含所有训练相关的内容
+ `test` 文件夹包含所有的我写的测试样例
+ `outputs` 文件夹包含模型输出的结果

### 目前实现的算法论文

+ [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://aclanthology.org/2021.emnlp-main.552.pdf)
  + [unsupervised SimCSE](examples/sentence_embedding/01_u_sim_cse.py)
  + [supervised SimCSE](examples/sentence_embedding/02_s_sim_cse.py)
+ [ESimCSE: Enhanced Sample Building Method for Contrastive Learning of Unsupervised Sentence Embedding](https://arxiv.org/pdf/2109.04380.pdf)
  + [ESimCSE](examples/sentence_embedding/03_e_sim_cse.py)
+ [RocketQA 系列文章](https://github.com/PaddlePaddle/RocketQA)
  + [RocketQA, PAIR, RocketQA V2](examples/RocketQA/README.md)

### 目前完成的笔记

### 写在最后

本项目借鉴了众多优秀的开源项目, 非常感谢各位大佬的开源, 也正是因为此, 我才决定开源这个项目, 以记录我的学习历程。

同时我在阅读别人代码时, 会发现很多代码有很多问题, 这在所难免。如果你发现我的代码有问题, 请一定要提 issue, 这是对我最大的帮助, 万分感谢 !!!
