
# Relation Extraction 关系抽取

使用百度的 DuIE 2.0 数据集进行 **关系三元组抽取** 的实验。

在 DuIE 2.0 中, 一共有 48 个关系标签, 我筛选掉训练集中样本数量低于 1000 的关系标签, 这样还剩下 34 个标签。

DuIE 2.0 的项目地址在: [千言](https://www.luge.ai/#/luge/dataDetail?id=5) 平台上有, 更多内容参考项目地址。

打算测试的算法有四个:

+ [x] CasRel
+ [x] PRGC
+ [x] TPLinker
+ [x] GPLinker
+ [x] OneRel

### 模型总结

##### CasRel 和 PRGC 模型

CasRel 和 PRGC 都属于 pipeline 管道模型, 训练方式是联合的, 预测方式是分阶段的, 问题主要有两个:

+ exposure bias: 训练和预测阶段的输入是不一致的
+ cascading errors: 上一个阶段预测错误, 下一个阶段的预测也是错误的

CasRel 模型的流程大致如下:

+ 采用 "序列标注" 的方式去预测 subject
+ 将 subject 的首尾词向量取平均, 作为 subject 向量
+ 将 subject 向量 "融入" 每一个词向量中
+ 每一个 relation 做一次 "序列标注", 去预测 object

会出现 exposure bias 的原因是在训练阶段, 选取的 subject 不是模型预测出来的 subject 位置, 而是数据集给出的 subject 位置。
在 data_collate 阶段比较麻烦的是需要针对某一个 subject 去统计所有可能的 relation 和 object 标签

PRGC 模型的流程大致如下:

+ 对一句话中所有的词向量进行池化, 生成句向量, 采用 "多标签分类" (二分类) 的方式预测一句话中所有 relation
+ 通过 `nn.Embedding` 层生成 relation 向量, 然后 "融入" 每一个向量中
+ 进行两次 "序列标注", 分别预测 subject 和 object 实体
+ 进行一次 token pairs 级别的 correspondence 预测, 如果 subject head 和 object head 之间预测为 1, 其它预测为 0
+ 在解码阶段, 先解码 subject 和 object 序列, 再根据 correspondence 矩阵将 subject 和 object 实体关联在一起

会出现 exposure bias 的原因是在训练阶段, 选取的 relation 不是模型预测出来的 relation, 而是数据集给出的 relation。
在 data_collate 阶段比较麻烦的是需要针对某一个 relation 去统计所有可能的 subject 和 object 标签

CasRel 和 PRGC 最大的不同在于对于 relation 的处理:

+ CasRel 是每一个 relation 一个 "序列标注" 的模型
+ PRGC 是给每一个 relation 一个向量编码, 然后所有的 relation 共享 "序列标注" 的模型

理解上面的内容很重要, 其它的一些区别是可以互换的, 比方说 CasRel 使用 span 的 "序列标注" 模型, PRGC 使用 BIO 的 "序列标注" 模型, 这两者是可以互换的

##### TPLinker, GPLinker 和 OneRel 模型

TPLinker, GPLiner 和 OneRel 都是基于 Token-Pairs 矩阵的联合模型, 即进行一次性预测, 再根据预测结果进行解码, 获得结果。下面分别介绍三个模型:

TPLinker 全称是 Token-Pairs Linker, 将关系抽取转化为 2N+1 个多分类任务:

+ 实体识别: 实体包括所有的 subject 和 object 实体, 所有 "关系" 共享一个实体识别任务, 记为 EH-ET, 由于 EH 一定小于或者等于 ET, 因此是二分类任务
+ head 位置识别: 识别所有的 (subject head, object head) 对, 一个 "关系" 一个分类任务, 记作 SH-OH, 是三分类任务:
  + 如果 SH 在 OH 左边或者重合, 标签为 1
  + 如果 SH 在 OH 右边, 标签为 2
  + 其它情况, 标签为 0
+ tail 位置识别: 识别所有的 (subject tail, object tail) 对, 一个 "关系" 一个分类任务, 记作 ST-OT, 是三分类任务, 方式和上面一样

解码流程如下:

1. 根据 EH-ET 的结果, 提取所有的实体, 并构建字典 D: key 值是 head 索引, value 值是所有的 tail 索引候选集
2. 根据 ST-OT 的结果, 提取所有可能的 (rl, st, ot) 元组, 构建 tail 候选集 S
3. 根据 SH-OH 的结果, 提取所有可能的 (rl, sh, oh) 元组, 并根据字典 D, 找到所有可能的 (rl, st, ot) 元组, 如果其出现在候选集 S 中, 则认为构成 (sh, st, rl, oh, ot) 五元组

GPLinker 和 TPLinker 很像, 将关系抽取变成了 2N+2 个二分类任务, 将原本 TPLinker 实体识别变成了 subject 识别和 object 识别, 使得解码变得简单, 解码流程如下:

1. 解码找到所有的 subject 实体和 object 实体
2. 遍历所有的 subject 和 object 实体对, 在遍历每一个标签 ~~(对, 三层循环)~~ , 如果此处的 (sh, rl, oh) 和 (st, rl, ot) 的标签都是 1, 那么认为其可以构成 (sh, st, rl, oh, ot) 五元组

OneRel 则强调其是一个分类任务, 不再是多个分类任务, 将关系抽取转化为 N 个四分类任务 (N 表示关系数, 也就是一个关系一个分类任务)。我们构建如下的 token-pairs 矩阵:

+ `dim=0` 表示 subject 索引, `dim=1` 表示 object 索引
+ `1` 标签表示 (subject-head, object-head) 关系
+ `2` 标签表示 (subject-head, object-tail) 关系
+ `3` 标签表示 (subject-tail, object-tail) 关系
+ 其它情况均是 `0` 标签

解码流程如下:

1. 遍历 token-pairs 矩阵, 找到所有 (subject-head, object-head) 关系
2. 对于每一个 (subject-head, object-head), 向右寻找距离最近的 (subject-head, object-tail), 找到后再向下寻找距离最近的 (subject-tail, object-tail), 如果都找到, 则关系命中
