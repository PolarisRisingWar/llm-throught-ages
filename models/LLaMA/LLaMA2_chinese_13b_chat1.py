#权重下载自https://huggingface.co/hiyouga/Llama-2-Chinese-13b-chat

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained("/data/pretrained_models/llama2-chinese-13b-chat",
                                             device_map='auto')
model.eval()
tokenizer=AutoTokenizer.from_pretrained("/data/pretrained_models/llama2-chinese-13b-chat",trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

template = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: {}\nAssistant: "
)

generate_ids = model.generate(tokenizer(template.format("晚上睡不着怎么办"), return_tensors='pt').input_ids.cuda(),
                              max_new_tokens=4096, streamer=streamer)