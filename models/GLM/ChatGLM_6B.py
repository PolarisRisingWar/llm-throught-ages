from transformers import AutoTokenizer, AutoModel

#checkpoint下载地址：https://huggingface.co/THUDM/chatglm-6b/tree/main
tokenizer = AutoTokenizer.from_pretrained("/data/wanghuijuan/pretrained_model/chatglm-6b",trust_remote_code=True)
model = AutoModel.from_pretrained("/data/wanghuijuan/pretrained_model/chatglm-6b",trust_remote_code=True).half().cuda()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
