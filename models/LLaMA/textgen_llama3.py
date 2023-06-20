#首先运行如下命令，将2部分checkpoint合并
#LLaMA的checkpoint权重下载自https://huggingface.co/shibing624/chinese-alpaca-plus-7b-hf
#Chinese-alpaca-lora-7b权重下载自https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b
#（据我猜测也可以直接下载这个模型库的checkpoint：https://huggingface.co/shibing624/chinese-alpaca-plus-7b-hf）

#python whj_download/githubs/textgen/textgen/gpt/merge_peft_adapter.py \
#    --base_model_name_or_path /data/whj/pretrained_model/llama-7b/llama \
#    --peft_model_path /data/whj/pretrained_model/chinese-alpaca-lora-7b \
#    --output_type huggingface \
#    --output_dir /data/whj/pretrained_model/llama_and_alpacalora_by_me \
#    --offload_dir /data/whj/cache

from textgen import GptModel

def generate_prompt(instruction):
  return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response:"""

model = GptModel("llama", "/data/whj/pretrained_model/llama_and_alpacalora_by_me")
predict_sentence = generate_prompt("问：用一句话描述地球为什么是独一无二的。\n答：")
r = model.predict([predict_sentence])
print(r)  # ['  地球上独一无二，是一个独特的星球。']