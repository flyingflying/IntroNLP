# Author: lqxu

import os
import sys

# used for HuggingFace Transformers Trainer API
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# used for HuggingFace Tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Python 默认只会把运行文件所在的目录加入 sys.path 中, 不会把工作目录加入 sys.path 中 (天坑啊 !!!)
sys.path.insert(0, "./")
