#源码相关部分：
#BartForSequenceClassification：https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1474
#BartModel：https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1172

import json
from datasets import Dataset,DatasetDict
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer

def convert2Dataset(split_name):
    """将数据集转换为datasets.Dataset的格式"""
    results={'text':[],'label':[]}
    original_data=json.load("文件")
    
    for sample in original_data:
        results['text'].append("文本")
        results["label"].append("数值形式的标签")

    return Dataset.from_dict(results)

dataset={}
dataset['train']=convert2Dataset('train')
dataset['valid']=convert2Dataset('valid')
dataset['test']=convert2Dataset('test')
dataset=DatasetDict(dataset)

tokenizer = AutoTokenizer.from_pretrained("/data/pretrained_model/bart-base-chinese")

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True,padding=True)

    model_inputs["labels"] = examples["label"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("/data/pretrained_model/bart-base-chinese",
                                                            num_labels=2)


#训练
training_args = TrainingArguments(output_dir="/data/cpt_output",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=32,
                                  save_steps=10,
                                  save_total_limit=3,
                                  num_train_epochs=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['valid']
)

trainer.train()

#测试
from transformers import TextClassificationPipeline
#如果在另一个Python文件中调用模型，需要重新做如下定义：
#model=AutoModelForSequenceClassification.from_pretrained("模型路径")
#tokenizer=AutoTokenizer.from_pretrained("/data/pretrained_model/bart-base-chinese")
classifier=TextClassificationPipeline(model=model,tokenizer=tokenizer,binary_output=True,return_token_type_ids=False)

#然后就可以直接通过classifier(句子)[0]["label"]得到标签了