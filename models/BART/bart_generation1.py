#代码参考了https://huggingface.co/docs/transformers/tasks/summarization
#另外还可参考https://blog.csdn.net/daotianweng/article/details/121036353

#训练阶段
import json,random,copy,os,re

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,\
                        Seq2SeqTrainer,EarlyStoppingCallback

from datasets import Dataset,DatasetDict

os.environ["WANDB_DISABLED"]="true"

# 参数设置
model_path="/data/wanghuijuan/pretrained_model/bart-base-chinese"
output_path="/data/wanghuijuan/my_checkpoints/output202308031930"
source_max_length=1024
target_max_length=128  #这2个都是模型限长

# Part 1: 导入数据集
def random_split_dataset(data_list:list,split_ratio:list=[8,1,1],random_seed:int=20230331):
    """输入数据集列表、划分比例和随机种子，返回划分后的数据集列表"""
    random.seed(random_seed)
    data_list_copy=copy.deepcopy(data_list)
    random.shuffle(data_list_copy)
    begin_index=0
    result=[]
    split_sum=sum(split_ratio)
    for split in split_ratio[:-1]:
        end_index=begin_index+int(split/split_sum*len(data_list_copy))
        result.append(data_list_copy[begin_index:end_index])
        begin_index=end_index
    result.append(data_list_copy[begin_index:])
    return result

random_seed=20230721
split_ratio=[8,1,1]

original_data=json.load(open("whj_code2/llm-throught-ages/models/BART/try_data2.json"))
datasets_indexes=random_split_dataset(list(range(len(original_data))),split_ratio,random_seed)

def convert2Dataset(split_number):
    """将数据集转换为datasets.Dataset的格式"""
    results={'text':[],'summary':[]}
    
    for sample in datasets_indexes[split_number]:
        results['text'].append(original_data[sample]["source"].strip())
        results["summary"].append(original_data[sample]["target"].strip())

    return Dataset.from_dict(results)

datasets={}
datasets['train']=convert2Dataset(0)
datasets['valid']=convert2Dataset(1)
datasets=DatasetDict(datasets)

# Part 2: 构建模型
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForSeq2SeqLM.from_pretrained(model_path)

def preprocess_function(examples):
    model_inputs = tokenizer(examples['text'],max_length=source_max_length,truncation=True)

    labels = tokenizer(text_target=examples["summary"],max_length=target_max_length,truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = datasets.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Part 4：训练

training_args = Seq2SeqTrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_strategy="epoch",
    save_steps=1,
    save_total_limit=3,
    num_train_epochs=100,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(10)],
)

trainer.train()

#测试阶段
def find_latest_checkpoint(path):
    # 获取指定路径下所有的子文件夹名
    subfolders = os.listdir(path)
    # 使用正则表达式匹配出文件夹名中的数字
    pattern = re.compile(r'checkpoint-(\d+)')
    # 存储数字和对应文件夹名
    num_folder_dict = {}
    for subfolder in subfolders:
        match = pattern.search(subfolder)
        if match:
            num_folder_dict[int(match.group(1))] = subfolder
    # 找出最大的数字对应的文件夹名
    latest_checkpoint_folder = num_folder_dict[max(num_folder_dict.keys())]
    return os.path.join(path, latest_checkpoint_folder)

final_checkpoint_folder=find_latest_checkpoint(output_path)
model=AutoModelForSeq2SeqLM.from_pretrained(final_checkpoint_folder)

for i in datasets_indexes[2]:
    outputs=model.generate(tokenizer.encode(original_data[i]["source"].strip(),return_tensors="pt",max_length=source_max_length),
                            max_new_tokens=target_max_length,do_sample=False)
    p=tokenizer.decode(outputs[0],skip_special_tokens=True).replace(" ","")
    #在这里对测试结果进行处理