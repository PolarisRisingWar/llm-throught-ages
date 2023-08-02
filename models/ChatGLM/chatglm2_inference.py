from transformers import AutoModel,AutoTokenizer

model_name="/data/wanghuijuan/pretrained_models/chatglm2-6b"

model = AutoModel.from_pretrained(model_name,
                                  load_in_8bit=False, 
                                  trust_remote_code=True, 
                                  device_map='auto')

tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = """math word problem问题：你是一位优秀的数学专家，擅长做数学题。

以下给出一些问答示例：
甲数除以乙数的商是1.5，如果甲数增加20，则甲数是乙的4倍．原来甲数=． -> x=20/(4-1.5)*1.5 -> 12
客车和货车分别从A、B两站同时相向开出，5小时后相遇．相遇后，两车仍按原速度前进，当它们相距196千米时，货车行了全程的80%，客车已行的路程与未行的路程比是3：2．求A、B两站间的路程． -> x=196/(80%+((3)/(3+2))-1) -> 490
图书角有书30本，第一天借出了(1/5)，第二天又还回5本，现在图书角有多少本书？ -> x=30*(1-(1/5))+5 -> 29

现在我们知道：
source

你的输出是：
"""

def get_prompt(text):
    return prompt.replace('source',text)

def predict(text,history=[]):
    response,return_history=model.chat(tokenizer,get_prompt(text),history=history,temperature=0.01)
    return response,return_history

p,_=predict("甲、乙两车同时从相距230千米的两地相向而行，3小时后两车还相距35千米．已知甲车每小时行48千米，乙车每小时行多少千米？")  #此处返回值的第二个元素是可以用以长篇对话的history，在本场景中不需要
print(p)
#输出完全就是瞎说