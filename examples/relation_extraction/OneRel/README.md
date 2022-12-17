
# OneRel PyTorch re-implementation

### Overview

+ 论文名: OneRel: Joint Entity and Relation Extraction with One Module in One Step
+ 论文地址: https://arxiv.org/pdf/2203.05412.pdf
+ 官方代码: https://github.com/ssnvxia/OneRel

### Relation Specific Horns Tagging

原论文中对于三元组的描述和本项目中的不一致, 对应关系如下:

+ head entity: subject
+ tail entity: object
+ begin: head index
+ end: tail index

为了统一, 这里依然用 SRO 的方式进行描述, 方便理解。

对于 token_pairs 矩阵来说, 我们定义 `dim=0` 是 subject 索引, `dim=1` 是 object 索引。

在 OneRel 中, 我们将每一个关系转换成一个四分类任务, 任务标签定义如下:

+ (subject_head, object_head) 位置标记为 `1` 标签
+ (subject_head, object_tail) 位置标记为 `2` 标签
+ (subject_tail, object_tail) 位置标记为 `3` 标签
+ 其它位置标记为 `0` 标签

这种标记方式乍看之下好像没有问题, 但是实际上会有很大问题, 那就是单字实体的问题。原论文给出的解决办法是添加特殊 token, 即将分词改成下面的步骤:

+ 首先使用 `BasicTokenizer`, 也就是 `WhiteSpaceTokenizer` 进行分词
+ 用 `WhiteSpaceTokenizer` 分词完成后, 在两个词之间加入特殊字符 `[unused1]`
+ 然后再对每一个词用 `WordPiece` 进行分词

这种分词方式对于英文来说没有问题, 但是对于中文来说问题很大, 因为没有办法按照空格进行分词, 如果直接在两个字符之间加入 `[unused1]` 特殊字符, 其显存开销会特别大

这里采用 [@sc-lj](https://github.com/sc-lj/RelationExtraction) 的方法, 对标签加以限制, 并在解码处解决问题:

+ 如果 object 是单字实体, 那么 oh = ot, 那么此时 sh-oh 和 sh-ot 位置是相同的, 我们取 sh-oh 的标签
+ 如果 subject 是单字实体, 那么 sh = st, 那么此时 sh-ot 和 st-ot 位置是相同的, 我们取 sh-ot 的标签

解码的大概思路如下:

遍历每一个 (subject_head, object_head) 位置, 固定 subject_head 值向右寻找 object_tail, 然后再固定 object_tail 值, 向下寻找 subject_tail 值

同时原代码使用了 `torch.where` 的特性少使用了一次循环, 具体的做法见 [code](scheme.py) 。

### Experiments

本实验的代码主要是参考官方代码, 按照我之前的框架重构完成, 并借鉴了 [@sc-lj](https://github.com/sc-lj/RelationExtraction) 中单字实体的处理方式。
本实验和原代码保持一致, 没有采用什么训练技巧 (没有使用 lr_scheduler, 也没有使用 focal loss), 如果想提升性能, 就需要尝试使用这些技巧。

数据集的使用和 **CasRel** 中保持一致, 实验结果如下:

```text
                    precision    recall  f1-score   support

                主演     0.7757    0.8863    0.8273      5411
                作者     0.7845    0.8490    0.8155      3709
                歌手     0.7017    0.7888    0.7427      2580
                导演     0.8030    0.8631    0.8320      2484
                父亲     0.6655    0.7042    0.6843      1819
             成立日期     0.7648    0.8219    0.7924      1741
                妻子     0.7197    0.7283    0.7240      1498
                丈夫     0.7222    0.7280    0.7251      1500
                国籍     0.6365    0.6935    0.6638      1442
                母亲     0.6556    0.7571    0.7027      1091
                作词     0.7754    0.8236    0.7988      1111
                作曲     0.6949    0.7746    0.7326      1047
             毕业院校     0.7730    0.8306    0.8008       992
             所属专辑     0.7015    0.7697    0.7340       916
               董事长     0.6897    0.7517    0.7193       878
                朝代     0.5279    0.6105    0.5662       760
                嘉宾     0.5928    0.7236    0.6517       662
             出品公司     0.5356    0.6372    0.5820       802
                编剧     0.6509    0.6935    0.6715       796
             上映时间     0.6637    0.7485    0.7036       688
                饰演     0.6960    0.7343    0.7146       636
                简称     0.7373    0.8081    0.7711       521
               主持人     0.6731    0.7624    0.7149       505
                配音     0.5526    0.5976    0.5742       492
                获奖     0.5054    0.5890    0.5440       399
               主题曲     0.6611    0.6992    0.6797       399
                校长     0.7842    0.8367    0.8096       343
             总部地点     0.4080    0.4915    0.4459       352
                主角     0.4566    0.5402    0.4949       224
               创始人     0.5759    0.6379    0.6053       232
                票房     0.6755    0.8095    0.7365       252
               制片人     0.5782    0.7794    0.6639       204
                 号     0.8520    0.9314    0.8899       204
                祖籍     0.7512    0.7947    0.7724       190

         micro avg     0.7091    0.7808    0.7432     36880
         macro avg     0.6689    0.7410    0.7026     36880
weighted macro avg     0.7106    0.7808    0.7437     36880
```

单个标签的统计结果如下:

```text
                    precision    recall  f1-score   support

                 1     0.7424    0.8175    0.7781     36879
                 2     0.7416    0.8171    0.7775     36569
                 3     0.7359    0.8115    0.7719     36791

         micro avg     0.7400    0.8154    0.7758    110239
         macro avg     0.7400    0.8154    0.7758    110239
weighted macro avg     0.7400    0.8154    0.7758    110239
```
