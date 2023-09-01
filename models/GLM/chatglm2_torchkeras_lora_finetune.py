#用torchkeras包微调ChatGLM2模型，用的是Lora
#仅支持单卡运行，大概需要20G+的显存
#代码参考https://github.com/lyhue1991/torchkeras/blob/master/examples/ChatGLM2_LoRA%E2%80%94%E2%80%94transformers.ipynb

import random,copy,json

import torch

from transformers import AutoConfig,AutoModel,AutoTokenizer

from peft import get_peft_model, LoraConfig, TaskType,PeftModel

from accelerate import Accelerator

from torchkeras import KerasModel



model_name="/data/wanghuijuan/pretrained_models/chatglm2-6b"
config=AutoConfig.from_pretrained(model_name,trust_remote_code=True,device_map='auto')
tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model=AutoModel.from_pretrained(model_name,trust_remote_code=True,load_in_8bit=False)

max_seq_length=2432

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

history=[(get_prompt("甲、乙两车同时从相距230千米的两地相向而行，3小时后两车还相距35千米．已知甲车每小时行48千米，乙车每小时行多少千米？"), 'x=(230-35)/3-48 -> 17')]

def predict(text):
    response,_=model.chat(tokenizer,get_prompt(text),history=history,temperature=0.01)
    return response

def preprocess(example):

    context=example["source"]
    target=example["target"]
    
    context_ids=tokenizer.encode(context,max_length=max_seq_length,truncation=True)
    
    target_ids = tokenizer.encode(target,add_special_tokens=False,max_length=max_seq_length,truncation=True)
    
    input_ids = context_ids + target_ids + [config.eos_token_id]
    
    # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss
    labels = [-100]*len(context_ids)+ target_ids + [config.eos_token_id]
    
    return {"input_ids": input_ids,
            "labels": labels,
            "context_len": len(context_ids),
            'target_len':len(target_ids)+1}

#original_data是所有数据的列表，每个元素是一个字典，包含source和target两个键值对
#datasets_indexes是一个列表，里面有3个列表，第一个列表是训练集的索引，第二个列表是验证集的索引

ds_train_token=[]
for i in datasets_indexes[0]:
    ds_train_token.append(preprocess(original_data[i]))

ds_valid_token=[]
for i in datasets_indexes[1]:
    ds_valid_token.append(preprocess(original_data[i]))



def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids) #之后按照batch中最长的input_ids进行padding
    
    input_ids = []
    labels_list = []
    
    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]
        
        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)
        
        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))
          
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

dl_train = torch.utils.data.DataLoader(ds_train_token,num_workers=2,batch_size=1,
                                       pin_memory=True,shuffle=True,
                                       collate_fn = data_collator)
dl_val = torch.utils.data.DataLoader(ds_valid_token,num_workers=2,batch_size=1,
                                    pin_memory=True,shuffle=True,
                                     collate_fn = data_collator)

model.supports_gradient_checkpointing = True  #节约cuda
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.is_parallelizable = True
model.model_parallel = True
model.print_trainable_parameters()



class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        
        #loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"],labels=batch["labels"]).loss

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_loss = self.accelerator.gather(loss).sum()
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics
    
KerasModel.StepRunner = StepRunner 


#仅仅保存lora可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator = None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)
    
def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path,'adapter_model.bin')),strict =False)
    self.from_scratch = False
    
KerasModel.save_ckpt = save_ckpt 
KerasModel.load_ckpt = load_ckpt 


keras_model = KerasModel(model,loss_fn = None,
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6))
ckpt_path = '/data/wanghuijuan/my_checkpoint/chatglm_gsm2k_ckpt'

keras_model.fit(train_data = dl_train,
                val_data = dl_val,
                epochs=50,patience=3,
                monitor='val_loss',mode='min',
                ckpt_path = ckpt_path,
                mixed_precision='fp16'
               )


model = PeftModel.from_pretrained(model,ckpt_path)
model = model.merge_and_unload()

print(predict("果园里有苹果树300棵，比桔树多20%，桔树有多少棵？"))
#标准答案应该是：
#x=300/(1+20%) -> 250