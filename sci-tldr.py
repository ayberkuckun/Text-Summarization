import argparse

import numpy as np
from nltk import sent_tokenize

import wandb
from datasets import load_dataset, load_from_disk, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel


def main(args):
    # model_checkpoint = "t5-small"
    # model_checkpoint = "facebook/bart-base"
    # model_checkpoint = "bert2gpt2"
    model_checkpoint = args.model_checkpoint

    if model_checkpoint == "bert2gpt2":
        encoder = "distilbert-base-uncased"
        decoder = "distilgpt2"

        bert2gpt2 = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)
        tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)

        bert2gpt2.config.decoder_start_token_id = tokenizer.cls_token_id
        bert2gpt2.config.eos_token_id = tokenizer.sep_token_id
        bert2gpt2.config.pad_token_id = tokenizer.pad_token_id
        bert2gpt2.config.vocab_size = bert2gpt2.config.encoder.vocab_size

        bert2gpt2.config.max_length = 20
        bert2gpt2.config.min_length = 0
        bert2gpt2.config.no_repeat_ngram_size = 3
        bert2gpt2.config.early_stopping = True
        bert2gpt2.config.length_penalty = 1.0
        bert2gpt2.config.num_beams = 4

        model = bert2gpt2

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge_score = load_metric("rouge")

    dataset_name = "scitldr"
    dataset_type = "AIC"
    dataset = load_dataset(dataset_name, dataset_type)

    def preprocess_function(examples):
        for i in range(len(examples["source"])):
            examples["source"][i] = "summarize: " + " ".join(examples["source"][i])
            examples["target"][i] = examples["target"][i][0]

        model_inputs = tokenizer(
            examples["source"], max_length=512, truncation=True
        )
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"], max_length=128, truncation=True
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

    # tokenized_datasets = load_from_disk("processed_dataset/")

    wandb.login()
    wandb.init(project="msi-TextSummarization", name=f"{model_checkpoint}-{dataset_name}")

    batch_size = 8
    num_train_epochs = 10

    args = Seq2SeqTrainingArguments(
        output_dir=f"results/{dataset_name}/{model_checkpoint}",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        report_to="wandb",
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        # load_best_model_at_end=True
    )

    bigger_train = concatenate_datasets([tokenized_datasets["train"], tokenized_datasets["validation"]])

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=bigger_train,
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', required=True, help="facebook/bart-base or t5-small or bert2gpt2")
    args = parser.parse_args()
    main(args)
