# THIS WAS OGIRINALLY TRAINED ON GOOGLE COLAB

# Colab things

# !pip install transformers[sentencepiece]
# !pip install datasets
# !pip install kaggle
# !pip install rouge_score
# !pip install nltk
# !pip install accelerate nvidia-ml-py3

import transformers
import os
import torch
from transformers import pipeline, AdamW, AutoTokenizer, AutoModel,  AutoModelForSeq2SeqLM, Trainer, get_scheduler, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric, DatasetDict
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from pynvml import *

# Colab stuff, import dataset from Reviews.csv file

# from google.colab import drive
# drive.mount('/content/gdrive')
# os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/kaggle"
# %cd /content/gdrive/"My Drive"/kaggle
# dataset_wine = load_dataset("csv", data_files="Reviews.csv", split="train")

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

# PRE-PROCESS NON-PADDED VERSION OF DATASET

def tru(s):
  return isinstance(s, str)

def tokenize_function(examples):
  tmp = np.array(examples["Summary"])
  vec = np.vectorize(tru)(tmp)
  if not np.all(vec):
    false_ind = np.where(vec == False)[0]
    for i in range(len(false_ind)):
      print("Replace None with empty string", examples["Summary"][false_ind[i]])
      examples["Summary"][false_ind[i]] = ""
  tokenized_dataset = tokenizer(examples["Text"], max_length=512, truncation=True)

  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["Summary"], max_length=128, truncation=True)

  tokenized_dataset["labels"] = labels["input_ids"]

  return tokenized_dataset

tokenized_datasets = dataset_wine.map(tokenize_function, batched=True)

metric = load_metric("rouge")
nltk.download('punkt')

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  # Replace -100 in the labels as we can't decode them.
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  
  # Rouge expects a newline after each sentence
  decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
  decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
  
  result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  # Extract a few results
  result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
  
  # Add mean generated length
  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)
  
  return {k: round(v, 4) for k, v in result.items()}

train_testvalid = tokenized_datasets.train_test_split(test_size=0.1)
valid_test = train_testvalid["test"].train_test_split(test_size=0.5)
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': valid_test['test'],
    'valid': valid_test['train']})
train_test_valid_dataset = train_test_valid_dataset.remove_columns(["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time", "Text", "Summary"])
tokenized_dataset = train_test_valid_dataset
train_test_valid_dataset

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(150000))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(10000))

model =  AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base').to("cuda")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

batch_size = 8
training_args = Seq2SeqTrainingArguments(
  output_dir="./results_32batchlowlr",
  evaluation_strategy="steps",
  learning_rate=2e-5,
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=batch_size,
  gradient_accumulation_steps=4,
  weight_decay=0.01,
  save_total_limit=3,
  num_train_epochs=1,
  predict_with_generate=True,
  #fp16=True,
)

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=small_train_dataset,
  eval_dataset=small_eval_dataset,
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics,
)

transformers.logging.set_verbosity_info()
trainer.train()