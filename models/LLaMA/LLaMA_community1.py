#权重下载来源：https://huggingface.co/huggyllama/llama-7b/tree/main

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/data/whj/pretrained_model/llama-7b/llama")

model = AutoModelForCausalLM.from_pretrained("/data/whj/pretrained_model/llama-7b/llama")