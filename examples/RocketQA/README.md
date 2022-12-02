
# RocketQA PyTorch re-implementation

百度开源的信息检索框架

其原代码在 [GitHub](https://github.com/PaddlePaddle/RocketQA) 上有, 但是是在飞桨框架下实现的, 这里给出 PyTorch 实现的主要代码。
这里使用的是 DuReader Retrieval 数据集, 由于实验需要非常多的 GPU 资源, 同时测试时间也非常的漫长, 因此无法将实验完全做完, 有条件的可以尝试完成, 没有条件的看看就好 ...

GPU 需求的资源多主要体现在以下一些方面:

+ 训练需求的显存大, 这主要体现在 RocketQA V2, batch size 为 1 也不一定能训练的起来
  + 在百度的源码中, MSMARCO 数据集的 passage 长度设置是 96, DuReader Retrieval 的 passage 长度是 384, 你可以尝试将其调小 (在 DuReader Retrieval 的论文中, 作者说明了尽量让文本长度大于 256, 文本长度调的太小可能会影响性能)
  + 在百度的源码中, MSMARCO 数据集的 list 大小为 128, NQ 数据集的 list 大小为 32, 你可以尝试调小 list 的大小, 这样也可以减轻数据增强部分的计算量
  + 可以改变 query 的计算方式, 详见代码注释, 性能可能也会下降 (因为训练变简单了)
+ 测试的运算量大, 由于 DPR 架构的特点, 每一次测试都需要对800万的 passage 文本进行句向量编码, 单张3090需要16个小时才能编码完成 (含分词时间和 faiss 索引建立时间)
+ 数据增强运算量也很大, 训练如果有90万个 query, 每一个 query 只对 top-50 进行清洗, 需要进行4500万次的句向量编码 
  + 百度给了 MSMARCO 和 NQ 数据集增强后的结果, 但是对于 DuReader Retrieval, 只给出了 negative passage (已经很人性化啦), 如果要跑数据增强, 一定要确保代码的正确性再跑, 或者分阶段跑 !!!

如果你有 4 张 3090, 并且有两个星期的时间, 在代码不出错的情况下, 应该可以完成整套 RocketQA 的实验

百度在 [PaddleNLP](https://gitee.com/paddlepaddle/PaddleNLP) 中有相关的集成, 采用 SimCSE 对训练好的 RocketQA query 编码器进行微调, 具体可以参考: [政务问答案例](https://gitee.com/paddlepaddle/PaddleNLP/tree/develop/applications/question_answering/faq_system)

相关论文地址:
+ [RocketQA v1](https://arxiv.org/abs/2010.08191)
+ [RocketQA v2](https://arxiv.org/abs/2110.07367)
+ [PAIR](https://aclanthology.org/2021.findings-acl.191/)
+ [DuReader Retrieval](https://arxiv.org/abs/2203.10232)

文件说明:
+ DuReader Retrieval 数据的预处理和分词: [code](du_reader_retrieval_utils.py)
+ RocketQA v2 模型架构和显存测试: [code](joint_train.py)
+ PAIR 模型架构: [code](pair.py)
+ RocketQA v1 dual encoder 完整的训练代码: [code](01_train_basic_dual_encoder.py)
+ RocketQA v1 dual encoder 生成测试文件的代码: [code](02_test_basic_dual_encoder.py)
+ 测试代码: [code](evaluation.py)

requirements:
```requirements.txt
faiss==1.7.2
torch==1.13.0
transformers==4.24.0
pytorch-lightning==1.8.2
```

RocketQA v1 dual encoder 运行方式:
```shell
python examples/RocketQA/01_train_basic_dual_encoder.py
python examples/RocketQA/02_test_basic_dual_encoder.py
python examples/RocketQA/evaluation.py
```

| Model        | MRR@10 | recall@1 | recall@50 |
|--------------|--------|----------|-----------|
| dual-encoder | 24.46  | 15.40    | 72.60     |

效果不好, 我认为的主要原因有:

原代码中 batch size 是 128, 我这里的 batch size 是 20, 差距有点大, 在对比学习中, 负样本的数量对模型的训练影响还是很大的

原代码中训练了 3 个 epochs, 我只训练了 2 个 epochs, 多训练一些效果肯能会好一些

改进方案:
  + 有能力的话增加 batch size, 并多训练一轮
  + 增加 eval 过程, 评测方式是计算 accuracy (multiple choice 是否选择正确), 并用其选择模型
