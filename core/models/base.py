# Author: lqxu

""" HuggingFace 模块库的初步封装, 只测试了 BertModel, 未来会使用 AutoModel 替换 BertModel """

import json
from typing import *

import torch
from torch import nn
from transformers import PretrainedConfig, BertPreTrainedModel, BertModel, BertConfig

__all__ = ["BaseConfig", "BaseModel"]

pretrained_short_name_dict = {
    "base": "hfl/chinese-bert-wwm-ext",
    "roberta": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base",
    "pert": "hfl/chinese-pert-base",
    "roberta-large": "hfl/chinese-roberta-wwm-ext-large",
    "pert-large": "hfl/chinese-pert-large",
    "macbert-large": "hfl/chinese-macbert-large"
}

default_pretrained = "roberta"


class BaseConfig(PretrainedConfig):

    pretrained_name: str
    bert_config: Optional[BertConfig]

    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)
        # 加载使用的预训练模型名称
        pretrained_name: str = kwargs.pop("pretrained_name", default_pretrained)
        self.pretrained_name = pretrained_short_name_dict.get(pretrained_name, pretrained_name)

        # SBERT 模型的配置文件
        bert_config = kwargs.pop("bert_config", None)
        if isinstance(bert_config, Dict):
            self.bert_config = BertConfig(**bert_config)
        elif isinstance(bert_config, BertConfig):
            self.bert_config = bert_config
        else:
            self.bert_config = None

        # PyTorch 版本
        self.torch_version: str = torch.__version__

    def to_json_string(self, use_diff: bool = True) -> str:
        # 加载配置文件
        config_dict = self.to_diff_dict() if use_diff else self.to_dict()

        # 处理 BERT 模型的配置文件
        bert_config = config_dict.get("bert_config", None)
        if isinstance(bert_config, BertConfig):
            config_dict["bert_config"] = bert_config.to_diff_dict() if use_diff else bert_config.to_dict()

        # 转换成 json 字串
        return json.dumps(config_dict, indent=4, sort_keys=True, ensure_ascii=False) + "\n"


class BaseModel(BertPreTrainedModel):
    config_class = BaseConfig

    def __init__(self, config: BaseConfig, bert_kwargs: Dict = None):
        super(BaseModel, self).__init__(config)

        if config.bert_config is None:
            if bert_kwargs is None:
                bert_kwargs = {}
            self.bert: BertModel = BertModel.from_pretrained(
                config.pretrained_name, add_pooling_layer=False, **bert_kwargs)
            config.bert_config = self.bert.config
        else:
            self.bert: BertModel = BertModel(config=config.bert_config, add_pooling_layer=False)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        super(BaseModel, self).resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        return super(BaseModel, self).get_position_embeddings()

    def _reorder_cache(self, past, beam_idx):
        super(BaseModel, self)._reorder_cache(past, beam_idx)
