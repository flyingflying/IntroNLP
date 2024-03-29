
# GPLinker For Event Extraction 

使用 **GPLinker** 实现事件抽取。相关资源信息如下:

+ 苏剑林的博客: [GPLinker：基于GlobalPointer的事件联合抽取](https://www.kexue.fm/archives/8926)
+ 苏剑林的代码: [GitHub](https://github.com/bojone/GPLinker/blob/main/duee_v1.py)

本实验没有使用 **稀疏版** 的多标签交叉熵作为损失函数, 而是直接使用 **原版的** 多标签交叉熵作为损失函数, 主要原因是一方面稀疏版的代码有待商榷, 另一方面感觉并不能使训练变快很多。

### Global Pointer 系列博客

+ [寻求一个光滑的最大值函数](https://www.kexue.fm/archives/3290)
+ [函数光滑化杂谈：不可导函数的可导逼近](https://www.kexue.fm/archives/6620)
+ [将“softmax+交叉熵”推广到多标签分类问题](https://www.kexue.fm/archives/7359)
+ [让研究人员绞尽脑汁的Transformer位置编码](https://www.kexue.fm/archives/8130)
+ [Transformer升级之路：2、博采众长的旋转式位置编码](https://www.kexue.fm/archives/8265)
+ [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://www.kexue.fm/archives/8373)
+ [Efficient GlobalPointer：少点参数，多点效果](https://www.kexue.fm/archives/8877)
+ [GPLinker：基于GlobalPointer的实体关系联合抽取](https://www.kexue.fm/archives/8888)
+ [GPLinker：基于GlobalPointer的事件联合抽取](https://www.kexue.fm/archives/8926)

### 简介

一个 **事件 (event)** 由一个 **触发词 (trigger)** 和若干 **论元 (argument)** 组成。无论是触发词, 还是论元, 都是文本中的 **片段 (span)** 。
因此我们可以将事件抽取转换成 NER 任务, 将触发词和论元当作 **命名实体 (named entity)** 来识别。

一个事件除了触发词和论元外, 其本身还有类型标签, 记作 **事件类型 (event type)** 。每一个论元也有类型标签, 记作 **论元角色 (argument role)** 。
因此在将论元转化成 NER 识别时, 其实体类型是由事件类型和论元角色组成的二元组。触发词没有标签, 我们可以给其一个固定的角色标签, 比方说："触发词" 。

解决 NER 的办法有很多, 这里使用的是 efficient global pointer, 并加上了旋转式位置编码。你也可以尝试用其它 NER 的方式来替换, 比方说 BIO 序列标注的方案等等。

如果一段文本中某种事件类型仅会出现一次, 那么上述方案已经解决问题了。但是往往一段文本中同类型的事件会出现很多次, 那么应该怎么解决呢?

借鉴 TPLinker 的方案, 我们用 Token-Pairs 的方式去识别实体的头和尾链接。对于一个事件中的任意两个实体 (论元/触发词), 我们将其头位置连接起来, 构成一个 heads 矩阵, 然后将尾位置连接起来, 构成一个 tails 矩阵。
和关系抽取不同的是, 在这里, 两个实体的连接是没有顺序的, 我们人为规定一个顺序: 始终从较小的位置往较大的位置连接。这样矩阵就只保留上三角, 下三角可以直接 mask 掉了。

那么怎么进行解码呢? 从上面的构造方案可知, 一个事件的所有实体在图中是两两相连的, 不同事件的实体之间至少存在一对实体是不相连的 (除非出现一个事件完全包含另一个事件的情况)。
那么, 此时解码就转化成找 **完全图 (completed graph)** 或者 **clique (团)** 的问题。大体的算法如下:

1. 将一句话中识别出来的所有实体按照事件类型进行聚类 (不同事件类型之间不会进行完全图的搜索), 聚类后的某一个集合 N 
2. 遍历集合 N 中的实体对 (n1, n2), 判断其是否相连, 如果不相连, 则分别寻找和 n1, n2 相连的所有实体, 并加入集合 N‘ 中。
值得一提的是, 这里是遍历所有的实体对, 如果遇到不相连的实体对, 不会中断循环。这样会导致集合 N‘ 中出现多个重复的子图, 因此需要对集合 G 中的子图进行去重。~~对, 你没有看错, 就是对 **子图(实体的集合)** 去重~~
3. 如果集合 N‘ 是空集, 说明集合 N 中的所有实体是两两相连的, 那么这些实体就构成一个事件, 加入事件集合 E 中; 如果集合 N‘ 不是空集, 那么遍历集合 N‘ 中所有的子图, 进行第二步操作 (此时第二步中的集合 N 就是集合 N‘ 中的某一个子图)。
4. 遍历事件 E, 如果事件 E 中不存在触发词实体, 则将其删除

好了, 这就是一个完整的事件抽取方案。如果对上述方法有不理解的地方, 可以参考： [code](scheme.py) 。需要说明的是, 这里是利用递归算法实现的解码算法, 为了稳定, 可以考虑自行变成循环的方式。

关于 heads 和 tails 的链接识别, 这里使用的是 efficient global pointer (没有旋转式位置编码), 你可以换成其它任意的 Token-Pairs 的方案, 包括但不限于: 多头选择 (concat), 双仿射, CLN 等等。

### 实验结果

本实验使用的数据集是百度的 [DuEE 数据集](https://www.luge.ai/#/luge/dataDetail?id=6) 。

比赛中使用的测评方式是 NER, 只计算论元的识别正确率, 即按照 (event_type, argument_role, argument) 的形式计算 P, R 和 F1 。 这种方式会导致结果值的偏高。
正常的实验结果大概只有 40% - 50% 的 F1 分数值; 而按照论元实体来计算, 结果 F1 分数值能达到 75% - 85% 。
另一方面, DuEE 的数据集规模非常小, 大约只有 1.1 万个训练样本 ~~但是测试集有 3.4 万个样本, 百度你觉得这样合理吗?~~ , 同时部分 (event_type, argument_role) 组合的样本数量是个位数, 因此很容易过拟合。

比赛中的计算方式和我代码中的 `analysis_argument_f1_score` 很相似, 但需要注意的是我的计算中包括了触发词和论元的实体识别, 而比赛中的并没有包含触发词, 仅仅是论元的实体识别。

实验结果如下:

测试集事件级别的 P, R & F1 值如下:

```text
                        precision    recall  f1-score   support

      财经/交易-出售/收购     0.2667    0.3478    0.3019        23
          财经/交易-跌停     0.7857    0.7333    0.7586        15
          财经/交易-加息     0.6667    0.6667    0.6667         3
          财经/交易-降价     0.3636    0.4000    0.3810        10
          财经/交易-降息     0.2500    0.2500    0.2500         4
          财经/交易-融资     0.1667    0.1875    0.1765        16
          财经/交易-上市     0.1429    0.1429    0.1429         7
          财经/交易-涨价     0.0000    0.0000    0.0000         5
          财经/交易-涨停     0.5000    0.5000    0.5000        28
           产品行为-发布     0.5120    0.5705    0.5397       149
           产品行为-获奖     0.2857    0.2500    0.2667        16
           产品行为-上映     0.6250    0.5882    0.6061        34
           产品行为-下架     0.2917    0.3043    0.2979        23
           产品行为-召回     0.3556    0.4444    0.3951        36
              交往-道歉     0.3600    0.5000    0.4186        18
              交往-点赞     0.4000    0.5455    0.4615        11
              交往-感谢     0.1111    0.1250    0.1176         8
              交往-会见     0.3000    0.3333    0.3158         9
              交往-探班     0.5556    0.4545    0.5000        11
           竞赛行为-夺冠     0.3750    0.3810    0.3780        63
           竞赛行为-晋级     0.2051    0.2222    0.2133        36
           竞赛行为-禁赛     0.4737    0.5000    0.4865        18
           竞赛行为-胜负     0.3432    0.3985    0.3688       261
           竞赛行为-退赛     0.5000    0.5556    0.5263        18
           竞赛行为-退役     0.8182    0.8182    0.8182        11
           人生-产子/女     0.3529    0.4000    0.3750        15
              人生-出轨     0.5000    0.7500    0.6000         4
              人生-订婚     0.3000    0.3333    0.3158         9
              人生-分手     0.4762    0.6250    0.5405        16
              人生-怀孕     0.2857    0.2500    0.2667         8
              人生-婚礼     0.1429    0.1667    0.1538         6
              人生-结婚     0.4634    0.4419    0.4524        43
              人生-离婚     0.6757    0.6757    0.6757        37
              人生-庆生     0.2381    0.3125    0.2703        16
              人生-求婚     0.6667    0.6000    0.6316        10
              人生-失联     0.2857    0.2857    0.2857        14
              人生-死亡     0.3740    0.4667    0.4153       105
           司法行为-罚款     0.5152    0.5862    0.5484        29
           司法行为-拘捕     0.5376    0.5952    0.5650        84
           司法行为-举报     0.3571    0.4167    0.3846        12
           司法行为-开庭     0.6111    0.7857    0.6875        14
           司法行为-立案     0.5556    0.5556    0.5556         9
           司法行为-起诉     0.2917    0.3684    0.3256        19
           司法行为-入狱     0.5000    0.5500    0.5238        20
           司法行为-约谈     0.7812    0.7812    0.7812        32
          灾害/意外-爆炸     0.3000    0.3000    0.3000        10
          灾害/意外-车祸     0.3750    0.4286    0.4000        35
          灾害/意外-地震     0.3333    0.3158    0.3243        19
          灾害/意外-洪灾     0.4000    0.2857    0.3333         7
          灾害/意外-起火     0.5357    0.5172    0.5263        29
       灾害/意外-坍/垮塌     0.3333    0.3636    0.3478        11
          灾害/意外-袭击     0.1333    0.1176    0.1250        17
          灾害/意外-坠机     0.3846    0.3846    0.3846        13
           组织关系-裁员     0.5500    0.5000    0.5238        22
         组织关系-辞/离职     0.5263    0.5634    0.5442        71
           组织关系-加盟     0.4565    0.4118    0.4330        51
           组织关系-解雇     0.3846    0.3846    0.3846        13
           组织关系-解散     0.6364    0.7000    0.6667        10
           组织关系-解约     0.2857    0.4000    0.3333         5
           组织关系-停职     0.4000    0.4000    0.4000        10
           组织关系-退出     0.4800    0.5455    0.5106        22
           组织行为-罢工     0.3333    0.3750    0.3529         8
           组织行为-闭幕     0.5556    0.5556    0.5556         9
           组织行为-开幕     0.4688    0.5172    0.4918        29
           组织行为-游行     0.3000    0.2727    0.2857        11

             micro avg     0.4255    0.4652    0.4444      1737
             macro avg     0.4113    0.4385    0.4225      1737
    weighted macro avg     0.4294    0.4652    0.4453      1737
```

三个子任务结果如下:

```text
                    precision    recall  f1-score   support

         arguments     0.7806    0.7711    0.7758    5208.0
              head     0.6881    0.7286    0.7078    6131.0
              tail     0.6970    0.7357    0.7158    6221.0

         micro avg     0.7174    0.7437    0.7303   17560.0
         macro avg     0.7219    0.7451    0.7331   17560.0
weighted macro avg     0.7187    0.7437    0.7308   17560.0
```

训练集事件级别的 P, R & F1 值如下:

```text
                        precision    recall  f1-score   support

      财经/交易-出售/收购     0.9091    0.9524    0.9302       189
          财经/交易-跌停     0.9810    0.9904    0.9856       104
          财经/交易-加息     0.5000    0.5000    0.5000        26
          财经/交易-降价     0.8734    0.8734    0.8734        79
          财经/交易-降息     0.7500    0.7500    0.7500        32
          财经/交易-融资     0.9147    0.9291    0.9219       127
          财经/交易-上市     0.8627    0.8627    0.8627        51
          财经/交易-涨价     0.6719    0.6719    0.6719        64
          财经/交易-涨停     0.9912    1.0000    0.9956       224
           产品行为-发布     0.9976    0.9984    0.9980      1231
           产品行为-获奖     0.8851    0.8973    0.8912       146
           产品行为-上映     0.9759    0.9792    0.9775       289
           产品行为-下架     0.8622    0.8711    0.8667       194
           产品行为-召回     0.9867    0.9933    0.9900       299
              交往-道歉     0.9671    0.9671    0.9671       152
              交往-点赞     0.7978    0.8068    0.8023        88
              交往-感谢     0.8852    0.9153    0.9000        59
              交往-会见     0.9714    0.9714    0.9714        70
              交往-探班     0.9412    0.9412    0.9412        68
           竞赛行为-夺冠     0.9889    0.9933    0.9911       450
           竞赛行为-晋级     0.9840    0.9840    0.9840       312
           竞赛行为-禁赛     0.9407    0.9338    0.9373       136
           竞赛行为-胜负     0.9153    0.9917    0.9519      1917
           竞赛行为-退赛     0.9710    0.9710    0.9710       138
           竞赛行为-退役     0.9796    0.9796    0.9796        98
           人生-产子/女     0.9541    0.9630    0.9585       108
              人生-出轨     0.6667    0.6667    0.6667        33
              人生-订婚     0.9375    0.9524    0.9449        63
              人生-分手     0.9431    0.9667    0.9547       120
              人生-怀孕     0.8615    0.8615    0.8615        65
              人生-婚礼     0.6667    0.6667    0.6667        60
              人生-结婚     0.9899    0.9899    0.9899       296
              人生-离婚     0.9926    0.9926    0.9926       271
              人生-庆生     0.9848    0.9848    0.9848       132
              人生-求婚     0.7692    0.7792    0.7742        77
              人生-失联     0.9266    0.9439    0.9352       107
              人生-死亡     0.9829    0.9853    0.9841       819
           司法行为-罚款     0.9589    0.9633    0.9611       218
           司法行为-拘捕     0.9929    0.9929    0.9929       706
           司法行为-举报     0.8614    0.9062    0.8832        96
           司法行为-开庭     0.9808    0.9808    0.9808       104
           司法行为-立案     1.0000    1.0000    1.0000        80
           司法行为-起诉     0.9310    0.9310    0.9310       174
           司法行为-入狱     0.9308    0.9367    0.9338       158
           司法行为-约谈     0.9962    0.9886    0.9924       263
          灾害/意外-爆炸     0.6944    0.6849    0.6897        73
          灾害/意外-车祸     0.9343    0.9441    0.9391       286
          灾害/意外-地震     0.8810    0.8672    0.8740       128
          灾害/意外-洪灾     0.7083    0.7083    0.7083        48
          灾害/意外-起火     0.8810    0.8894    0.8852       208
       灾害/意外-坍/垮塌     0.7342    0.7342    0.7342        79
          灾害/意外-袭击     0.7197    0.7787    0.7480       122
          灾害/意外-坠机     0.8056    0.7982    0.8018       109
           组织关系-裁员     0.9236    0.9236    0.9236       144
        组织关系-辞/离职     0.9916    0.9950    0.9933       595
           组织关系-加盟     0.9857    0.9885    0.9871       349
           组织关系-解雇     0.7653    0.8242    0.7937        91
           组织关系-解散     0.9753    0.9875    0.9814        80
           组织关系-解约     0.3404    0.3556    0.3478        45
           组织关系-停职     0.8916    0.8605    0.8757        86
           组织关系-退出     0.9585    0.9686    0.9635       191
           组织行为-罢工     0.7500    0.7612    0.7556        67
           组织行为-闭幕     0.9153    0.9153    0.9153        59
           组织行为-开幕     0.9876    0.9876    0.9876       242
           组织行为-游行     0.7887    0.7887    0.7887        71

             micro avg     0.9383    0.9539    0.9460     13566
             macro avg     0.8841    0.8914    0.8876     13566
    weighted macro avg     0.9389    0.9539    0.9461     13566
```

三个子任务上的结果如下:

```text
                    precision    recall  f1-score   support

         arguments     0.9929    0.9828    0.9878   41172.0
              head     0.9987    0.9994    0.9990   48799.0
              tail     0.9992    0.9998    0.9995   49373.0

         micro avg     0.9972    0.9946    0.9959  139344.0
         macro avg     0.9969    0.9940    0.9954  139344.0
weighted macro avg     0.9971    0.9946    0.9959  139344.0
```

总结: 确实严重的过拟合, 主要问题还是训练集的样本太少了 (只有 1 万个训练样本 ...)
