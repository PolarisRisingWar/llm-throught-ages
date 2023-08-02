#直接在百川13B上进行推理
#文件下载自https://huggingface.co/baichuan-inc/Baichuan-13B-Chat

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("/data/wanghuijuan/pretrained_models/baichuan-13b", use_fast=False,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/data/wanghuijuan/pretrained_models/baichuan-13b", device_map="auto",
                                             torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/data/wanghuijuan/pretrained_models/baichuan-13b")
messages = []
messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
response = model.chat(tokenizer, messages)
print(response)
#输出示例：世界上第二高的山峰是乔戈里峰(K2)，位于巴基斯坦和中国边境的喜马拉雅山脉。它的高度为8,611米(28,251英尺)。