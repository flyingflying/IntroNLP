# Author: lqxu

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.insert(0, "./")

print("HuggingFace Hub 缓存地址:", os.environ["HUGGINGFACE_HUB_CACHE"])
print("HuggingFace Transformers 缓存地址:", os.environ["TRANSFORMERS_CACHE"])
print("HuggingFace Datasets 缓存地址:", os.environ["TRANSFORMERS_CACHE"])
