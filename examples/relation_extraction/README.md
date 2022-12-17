
### Relation Extraction 关系抽取

使用百度的 DuIE 2.0 数据集进行 **关系三元组抽取** 的实验。

在 DuIE 2.0 中, 一共有 48 个关系标签, 我筛选掉训练集中样本数量低于 1000 的关系标签, 这样还剩下 34 个标签。

DuIE 2.0 的项目地址在: [千言](https://www.luge.ai/#/luge/dataDetail?id=5) 平台上有, 更多内容参考项目地址。

打算测试的算法有六个:

+ [x] CasRel
+ [ ] PURE / PL-Marker
+ [x] TPLinker
+ [ ] GPLinker
+ [ ] PRGC
+ [x] OneRel MRC
