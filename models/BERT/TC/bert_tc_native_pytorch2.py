import csv
from tqdm import tqdm
from copy import deepcopy

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer,AutoModelForSequenceClassification

"""超参设置"""
pretrained_path='/data/pretrained_model/bert-base-chinese'
dropout_rate=0.1
max_epoch_num=16
cuda_device='cuda:2'
output_dim=2

"""加载数据集"""
#数据集都是CSV格式的文件
#label列是标签（这里是数字形式。如果是文本形式，要先映射为数字）， review列是要被分类的文本

#训练集
with open('chn_train.csv') as f:
    reader=csv.reader(f)
    header=next(reader)  #表头
    train_data=[[int(row[0]),row[1]] for row in reader]

#验证集
with open('chn_valid.csv') as f:
    reader=csv.reader(f)
    header=next(reader)
    valid_data=[[int(row[0]),row[1]] for row in reader]

#测试集
with open('chn_test.csv') as f:
    reader=csv.reader(f)
    header=next(reader)
    test_data=[[int(row[0]),row[1]] for row in reader]

tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
def collate_fn(batch):
    pt_batch=tokenizer([x[1] for x in batch],padding=True,truncation=True,max_length=512,return_tensors='pt')
    return {'input_ids':pt_batch['input_ids'],'token_type_ids':pt_batch['token_type_ids'],'attention_mask':pt_batch['attention_mask'],
            'label':torch.tensor([x[0] for x in batch])}

train_dataloader=DataLoader(train_data,batch_size=16,shuffle=True,collate_fn=collate_fn)
valid_dataloader=DataLoader(valid_data,batch_size=128,shuffle=False,collate_fn=collate_fn)
test_dataloader=DataLoader(test_data,batch_size=128,shuffle=False,collate_fn=collate_fn)

"""建模"""
#API文档：https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained(pretrained_path,num_labels=output_dim)
model.to(cuda_device)

"""构建优化器、损失函数等"""
optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-5)
loss_func=nn.CrossEntropyLoss()
max_valid_f1=0
best_model={}

"""训练与验证"""
for e in tqdm(range(max_epoch_num)):
    for batch in train_dataloader:
        model.train()
        optimizer.zero_grad()
        input_ids=batch['input_ids'].to(cuda_device)
        token_type_ids=batch['token_type_ids'].to(cuda_device)
        attention_mask=batch['attention_mask'].to(cuda_device)
        labels=batch['label'].to(cuda_device)
        outputs=model(input_ids,token_type_ids,attention_mask,labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    #验证
    with torch.no_grad():
        model.eval()
        labels=[]
        predicts=[]
        for batch in valid_dataloader:
            input_ids=batch['input_ids'].to(cuda_device)
            token_type_ids=batch['token_type_ids'].to(cuda_device)
            attention_mask=batch['attention_mask'].to(cuda_device)
            outputs=model(input_ids,token_type_ids,attention_mask)
            labels.extend([i.item() for i in batch['label']])
            predicts.extend([i.item() for i in torch.argmax(outputs.logits,1)])
        f1=f1_score(labels,predicts,average='macro')
        if f1>max_valid_f1:
            best_model=deepcopy(model.state_dict())
            max_valid_f1=f1

"""测试"""
model.load_state_dict(best_model)
with torch.no_grad():
    model.eval()
    labels=[]
    predicts=[]
    for batch in test_dataloader:
        input_ids=batch['input_ids'].to(cuda_device)
        token_type_ids=batch['token_type_ids'].to(cuda_device)
        attention_mask=batch['attention_mask'].to(cuda_device)
        outputs=model(input_ids,token_type_ids,attention_mask)
        labels.extend([i.item() for i in batch['label']])
        predicts.extend([i.item() for i in torch.argmax(outputs.logits,1)])
    print(accuracy_score(labels,predicts))
    print(precision_score(labels,predicts,average='macro'))
    print(recall_score(labels,predicts,average='macro'))
    print(f1_score(labels,predicts,average='macro'))

