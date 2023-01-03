# Author: lqxu

from torch import nn

from core.models import BaseConfig, BaseModel
from core.modules import EfficientGlobalPointer


class GPLinkerEEConfig(BaseConfig):
    def __init__(self, n_argument_labels: int, head_size: int = 64, dropout: float = 0.3, **kwargs):
        super(GPLinkerEEConfig, self).__init__(**kwargs)

        self.n_argument_labels = n_argument_labels

        self.dropout = dropout

        self.head_size = head_size


class GPLinkerEEModel(BaseModel):
    def __init__(self, config: GPLinkerEEConfig):
        super(GPLinkerEEModel, self).__init__(config)

        self.dropout = nn.Dropout(config.dropout)
        self.argument_classifier = EfficientGlobalPointer(
            config=config.bert_config, num_tags=config.n_argument_labels,
            head_size=config.head_size, use_rope=True
        )
        self.head_classifier = EfficientGlobalPointer(
            config=config.bert_config, num_tags=1,
            head_size=config.head_size, use_rope=False
        )
        self.tail_classifier = EfficientGlobalPointer(
            config=config.bert_config, num_tags=1,
            head_size=config.head_size, use_rope=False
        )
