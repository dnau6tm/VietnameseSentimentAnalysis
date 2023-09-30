import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
from typing import List
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from metrics import compute_metrics
from processdata import tokenization


model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base")

data_files = {"train": "train.csv", "test": "test.csv"}
train_dataset = load_dataset("./", split="train")
val_dataset = load_dataset("./", split="test")



train_dataset = train_dataset.map(tokenization, batched = True, batch_size = len(train_dataset))
val_dataset = val_dataset.map(tokenization, batched = True, batch_size = len(val_dataset))

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
  output_dir='./results',
  num_train_epochs=3,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=16,
  remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained('./out_model')


