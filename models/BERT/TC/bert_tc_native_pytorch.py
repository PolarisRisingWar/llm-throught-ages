#基于PyTorch 1.8.1，transformers 4.18.0，datasets 2构建

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import datasets
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_scheduler

dataset=datasets.load_from_disk("datasets/yelp_full_review_disk")

tokenizer = AutoTokenizer.from_pretrained("pretrained_models/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",truncation=True,max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#Postprocess dataset
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
#删除模型不用的text列

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#改名label列为labels，因为AutoModelForSequenceClassification的入参键名为label
#我不知道为什么dataset直接叫label就可以啦……

tokenized_datasets.set_format("torch")  #将值转换为torch.Tensor对象

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

model=AutoModelForSequenceClassification.from_pretrained\
                        ("pretrained_models/bert-base-cased",num_labels=5)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric=datasets.load_metric('datasets/accuracy.py')
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
