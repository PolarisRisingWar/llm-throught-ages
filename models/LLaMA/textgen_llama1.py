#LLaMA的checkpoint权重下载自https://huggingface.co/shibing624/chinese-alpaca-plus-7b-hf

from textgen import GptModel

def generate_prompt(instruction):
  return f"""Below is math word problem. Write an answer that appropriately solve this problem.
  
  ### Instruction:{instruction}
  
  ### Answer:"""

model=GptModel("llama","/data/whj/pretrained_model/llama-7b/llama")
predict_sentence = generate_prompt("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
r = model.predict([predict_sentence])
print(r)
#答案应该是48+48/2=72，反正事实上返回了一个相当混乱的答案
# GSM8K给的标准答案是：Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72