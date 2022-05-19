import argparse

import numpy as np
from nltk import sent_tokenize

import wandb
from datasets import load_dataset, load_from_disk, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer


def main():
    model_checkpoint = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge_score = load_metric("rouge")

    dataset_name = "billsum"
    dataset = load_dataset(dataset_name)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["text"], max_length=512, truncation=True
        )
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"], max_length=128, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
    tokenized_datasets.save_to_disk(f"processed_dataset/{dataset_name}/")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result_f = {"f1_" + key: value.mid.fmeasure * 100 for key, value in result.items()}
        result_p = {"p_" + key: value.mid.precision * 100 for key, value in result.items()}
        result_r = {"r_" + key: value.mid.recall * 100 for key, value in result.items()}

        result = {**result_f, **result_p, **result_r}

        return {k: round(v, 4) for k, v in result.items()}

    # tokenized_datasets = load_from_disk(f"processed_dataset/{dataset_name}/")

    wandb.login()
    wandb.init(project="TextSummarization", name=f"bart-{dataset_name}")

    batch_size = 8
    num_train_epochs = 5

    args = Seq2SeqTrainingArguments(
        output_dir=f"results/{dataset_name}/",
        evaluation_strategy="no",
        # evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_strategy="steps",
        save_strategy="epoch",
        fp16=True,
        report_to="wandb",
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        # load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
