#LLaMA的checkpoint权重下载自https://huggingface.co/shibing624/chinese-alpaca-plus-7b-hf
#Chinese-alpaca-lora-7b权重下载自https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b

from textgen import GptModel

def generate_prompt(instruction):
  return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response:"""

model = GptModel("llama", "/data/whj/pretrained_model/llama-7b/llama", peft_name="/data/whj/pretrained_model/chinese-alpaca-lora-7b")
predict_sentence = generate_prompt("问：用一句话描述地球为什么是独一无二的。\n答：")
r = model.predict([predict_sentence])
print(r)  # ['  我们只有一个地球，它唯一不可复制可重复造物。']