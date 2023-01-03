
# DuIE 2.0 

千言项目官网: https://www.luge.ai/#/luge/dataDetail?id=5

### DuIE 2.0 数据集完整介绍

一般的关系抽取是以三元组的形式呈现的, 即: (subject, predicate, object) 的形式。
其中, subject 和 object 是句子中的实体 (entity), 也可以理解成句子中的片段 (span)。
而 predicate 则是关系, 也可以用 relation 来代替, 其可以理解成标签, 在句子中一般是不出现的, 由人事先定义好。
对于要求更高的关系抽取任务来说, 每一个 subject 和 object 都是有类型的, 单独拿出来和 NER 没有区别。
在 DuIE2.0 数据集中, 其难度进一步提升, 对于一组关系来说, 其 object 值不一定只有一个, 还可能有多个。

具体的例子如下:

```json
{
  "text": "王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人",
  "spo_list": [
    {
      "predicate": "配音",
      
      "subject": "王雪纯",
      "subject_type": "娱乐人物",
      
      "object": {"@value": "晴雯", "inWork": "红楼梦"},
      "object_type": {"@value": "人物", "inWork": "影视作品"}
    },
    {
      "predicate": "主持人",
      
      "subject": "正大综艺",
      "subject_type": "电视综艺",
      
      "object": {"@value": "王雪纯"},
      "object_type": {"@value": "人物"}
    }
  ]
}
```

对于第一组关系来说: 
+ 关系标签是 "配音"
+ subject 实体是 "王雪纯", 标签是 "娱乐人物"
+ object 有两个实体, 一个是 "晴雯", 一个是 "红楼梦", 标签分别是 "人物" 和 "影视作品"
+ object 的两个实体是有关联的, 其中 "晴雯" 是主要实体, "红楼梦" 是附加实体, 和主要实体之间的关系是 inWork (参演的作品名)

再看另一个例子:

```json
{
  "text": "也长的很漂亮做了十多年赵薇的粉丝在这里表示永远爱她2007年，获得第5届MTV超级盛典内地最具风格女歌手奖", 
  "spo_list": [
    {
      "predicate": "获奖", 
      
      "object": {"onDate": "2007年", "@value": "MTV超级盛典内地最具风格女歌手", "period": "5"}, 
      "object_type": {"onDate": "Date", "@value": "奖项", "period": "Number"},
      
      "subject": "赵薇",
      "subject_type": "娱乐人物"
    }
  ]
}
```

对于第一组关系来说:
+ object 的主要实体是 "MTV超级盛典内地最具风格女歌手", 标签是 "奖项"
+ object 的附加实体有两个, 分别是 "2007年" 和 5, 对应的标签是 "Date" (日期) 和 "Number" (数字), 和主要实体之间的关系是 "onDate" (获奖日期) 和 "period" (第几届)
+ 关系标签是 "获奖"
+ subject 实体是 "赵薇", 类型是 "娱乐人物"

总的来说, DuIE2.0 数据集的样式是: SRO 三元组 + 实体类型 + 多个 object 值

### DuIE2.0 Baseline

怎么实现完整版的 DuIE2.0 任务呢? 即 SRO 三元组 + 实体类型 + 多个 object 的识别。百度官方 baseline 给出的方法就是 **拆** !!! 一共分成两步拆:

首先, 一组关系可能包含多个 object? 那么就拆, 有多少个 object 就拆成多少组关系。

比方说上面有这样一组关系: (王雪纯, 娱乐人物, 配音, (晴雯, 红楼梦), (人物, 影视作品), (@value, inWork))

现在拆成两组关系:
+ (王雪纯, 娱乐人物, 配音_@value, 晴雯, 人物)
+ (王雪纯, 娱乐人物, 配音_inWork, 红楼梦, 影视作品)

解码的时候, 只需要将他们拼在一起就好啦! 现在的问题变成了怎么进行 SRO 三元组 + 实体识别的信息抽取任务。

可以用上面所说的 PURE。这里百度提供了另一种思路, 那就是继续拆 !!!

现在将上面的两组关系拆成:

+ (subject_配音_@value_娱乐人物, 王雪纯)
+ (object_配音_@value_人物, 晴雯)
+ (subject_配音_inWork_娱乐人物, 王雪纯)
+ (object_配音_inWork_影视作品, 红楼梦)

由于在 DuIE2.0 的 scheme 中, 所有的 predicate 都是唯一的, 因此上面的实体标签可以去掉, 就变成了:

+ (subject_配音_@value, 王雪纯)
+ (object_配音_@value, 晴雯)
+ (subject_配音_inWork, 王雪纯)
+ (object_配音_inWork, 红楼梦)

此时已经不是 SRO 三元组问题了, 而是 NER 问题了, 直接 BIO 识别就好啦~ 最终解码阶段只要拼接在一起就可以得到结果。

如果一共有 55 种关系, 那么此时就是 110 个 BIO 序列标注问题, 也就是 110 个三分类问题。

百度对问题还进行了转化, 让这 110 个 BIO 序列标注任务共享 I 和 O 任务, 将 110 个三分类问题变成 110 + 2 个二分类问题。这就是他文档种所说的方法。

最后额外的说一句, 在 DuIE2.0 的数据集中, 由于所有的 predicate 在 scheme 中都是唯一的, 导致在知道 predicate 的情况下, subject 和 object 的类型就已经可以确定了, 那么就不存在识别 subject 和 object 类型的问题了。
这样问题就变成了: SRO 三元组 + 多个 object 的识别问题了。
