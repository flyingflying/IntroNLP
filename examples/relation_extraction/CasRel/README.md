
### CasRel PyTorch re-implementation

CasRel 是苏剑林大神和吉林大学一起发表的 **三元组** 关系抽取方案。相关的资源信息如下:

+ 论文名: A Novel Cascade Binary Tagging Framework for Relational Triple Extraction
+ 论文地址: https://aclanthology.org/2020.acl-main.136/
+ arxiv 论文地址: https://arxiv.org/abs/1909.03227
+ GitHub 地址: https://github.com/weizhepei/CasRel
+ 苏剑林博客: https://kexue.fm/archives/6671
+ bert4keras 实现: https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction.py

本实验的复现主要参考 [@Onion12138](https://github.com/Onion12138/CasRelPyTorch) 的代码和 bert4keras 中的代码, 感谢大佬们的开源 !!!

使用的数据集是 [DuIE2.0](https://www.luge.ai/#/luge/dataDetail?id=5) , 感谢百度的开源 !!!

由于 DuIE2.0 中部分关系的标签数量太少, 我们选取训练集中标签数量大于 1000 的 34 个标签进行实验。实验结果如下:

```text
                    precision    recall  f1-score   support

                主演     0.7910    0.8695    0.8284      5411
                作者     0.8392    0.8037    0.8211      3709
                歌手     0.7422    0.7310    0.7366      2580
                导演     0.8077    0.8454    0.8261      2484
                父亲     0.6585    0.6668    0.6627      1819
             成立日期     0.7916    0.8053    0.7984      1741
                妻子     0.6970    0.7356    0.7158      1498
                丈夫     0.6939    0.7373    0.7149      1500
                国籍     0.7404    0.6664    0.7015      1442
                母亲     0.6405    0.7397    0.6865      1091
                作词     0.8104    0.8002    0.8053      1111
                作曲     0.7539    0.7784    0.7660      1047
             毕业院校     0.8139    0.8246    0.8192       992
             所属专辑     0.7503    0.7544    0.7523       916
               董事长     0.7131    0.7107    0.7119       878
                朝代     0.5881    0.5711    0.5794       760
                嘉宾     0.6068    0.7039    0.6517       662
             出品公司     0.6062    0.6047    0.6055       802
                编剧     0.6837    0.6872    0.6855       796
             上映时间     0.7077    0.7180    0.7128       688
                饰演     0.6493    0.7248    0.6850       636
                简称     0.7609    0.7697    0.7653       521
               主持人     0.7036    0.7287    0.7160       505
                配音     0.5347    0.5488    0.5416       492
                获奖     0.4960    0.6140    0.5487       399
               主题曲     0.7222    0.6190    0.6667       399
                校长     0.8035    0.7988    0.8012       343
             总部地点     0.5439    0.4403    0.4867       352
                主角     0.5190    0.4866    0.5023       224
               创始人     0.5536    0.6681    0.6055       232
                票房     0.6138    0.7063    0.6568       252
               制片人     0.6558    0.6912    0.6730       204
                 号     0.8654    0.8824    0.8738       204
                祖籍     0.7884    0.7842    0.7863       190

         micro avg     0.7353    0.7554    0.7452     36880
         macro avg     0.6955    0.7123    0.7027     36880
weighted macro avg     0.7364    0.7554    0.7450     36880
```

只训练了 10 个 epochs, 效果还呈上升状态, 增加训练的 epochs 效果应该会更好  !!!