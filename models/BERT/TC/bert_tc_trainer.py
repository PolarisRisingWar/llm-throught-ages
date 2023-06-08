"""
   Copyright (c) [2023] [Wang Huijuan]
   [LLM Throughtout Ages] is licensed under Mulan PSL v2.
   You can use this software according to the terms and conditions of the Mulan PSL v2. 
   You may obtain a copy of Mulan PSL v2 at:
               http://license.coscl.org.cn/MulanPSL2 
   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
   See the Mulan PSL v2 for more details.  
   
基于PyTorch 1.8.1，transformers 4.18.0，datasets 2构建
"""

import datasets
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer

dataset=datasets.load_from_disk("datasets/yelp_full_review_disk")

tokenizer = AutoTokenizer.from_pretrained("pretrained_models/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"],padding="max_length",truncation=True,max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained("pretrained_models/bert-base-cased",
                                                            num_labels=5)

training_args = TrainingArguments(output_dir="pt_save_pretrained",evaluation_strategy="epoch")

metric=datasets.load_metric('datasets/accuracy.py')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
