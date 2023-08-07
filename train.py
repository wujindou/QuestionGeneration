#coding:utf-8
import numpy as np
import random
import torch
from transformers import set_seed
set_seed(42)
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

import os
os.environ["WANDB_DISABLED"] = "true"
checkpoint='fnlp/bart-base-chinese'

from transformers import AutoTokenizer
# from modeling_cpt import CPTForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_fast=False)
def preprocess_function(example):
  model_inputs = tokenizer(example['context'], truncation=True, padding="max_length", max_length=1024)
  labels = tokenizer(text_target=example["question"], max_length=128, truncation=True)
  model_inputs['labels'] = labels['input_ids']
  return model_inputs


import json 
train_data =[json.loads(line.strip()) for line in open('train.json','r',encoding='utf-8').readlines()]
dev_data = [json.loads(line.strip()) for line in open('dev.json','r',encoding='utf-8').readlines()]

import pandas as pd
import datasets
train_dataset =datasets.Dataset.from_pandas(pd.DataFrame(train_data))
dev_dataset = datasets.Dataset.from_pandas(pd.DataFrame(dev_data))


ds={'train':train_dataset,'validation':dev_dataset}
for split in ds:
  ds[split] = ds[split].map(preprocess_function)


from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

from rouge import Rouge
rouge_score = Rouge()
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(decoded_preds[0]+'\t'+decoded_labels[0])
    decoded_labels=['no' if t.strip()=='' else t for t in decoded_labels]
    decoded_preds=['no' if t.strip()=='' else t for t in decoded_preds]
    scores = rouge_score.get_scores(decoded_preds,decoded_labels,avg=True)
    for key in scores:
      scores[key] = scores[key]['f']*100
    result = scores
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from modeling_cpt import CPTForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

def get_model():
    # get_model is used for the model_init argument for trainer. This ensures reproducibility. Otherwise, weights from classification head are randomly initialized.
    # see https://discuss.huggingface.co/t/fixing-the-random-seed-in-the-trainer-does-not-produce-the-same-results-across-runs/3442
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint) 
    return model

# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint) 
training_args = Seq2SeqTrainingArguments(
    output_dir="bart_seq2seq_task9",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
    save_strategy='steps',
    eval_steps=20,
    save_steps=20,
    seed=42
)

trainer = Seq2SeqTrainer(
    model=get_model(),
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('./bart_seq2seq_qg')
