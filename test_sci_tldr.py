import argparse

import numpy as np
from nltk import word_tokenize
from transformers import pipeline, EncoderDecoderModel, AutoTokenizer
from datasets import load_dataset


def main(args):
    def process(examples):
        for i in range(len(examples["source"])):
            examples["source"][i] = " ".join(examples["source"][i])
            examples["target"][i] = examples["target"][i][0]

        return examples

    model_type = "bart"
    dataset_name = "scitldr"
    dataset_type = "AIC"
    dataset = load_dataset(dataset_name, dataset_type)
    processed_datasets = dataset.map(process, batched=True)

    model_checkpoint = args.model_checkpoint
    # model_checkpoint = f"results/{dataset_name}/{model_type}/{dataset_type}/checkpoint-810"

    if model_checkpoint == "bert2gpt2":
        model = EncoderDecoderModel.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

        def print_summary(idx):
            if dataset_type == "AIC":
                truncated_sentence = " ".join(word_tokenize(processed_datasets["test"]["source"][idx])[:512])
                inputs = tokenizer.encode(truncated_sentence, return_tensors="pt", max_length=512, truncation=True)
                outputs = model.generate(inputs, max_length=128)
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f""">>> Text: {truncated_sentence}""")

            else:
                # summary = summarizer(processed_datasets["test"]["source"][idx])[0]["summary_text"]
                print(f""">>> Text: {processed_datasets["test"]["source"][idx]}""")

            print(f"""\n>>> Actual Summary: {processed_datasets["test"]["target"][idx]}""")
            print(f"\n>>> Predicted Summary: {summary}")

    else:
        summarizer = pipeline("summarization", model=model_checkpoint)

        def print_summary(idx):
            if dataset_type == "AIC":
                truncated_sentence = " ".join(word_tokenize(processed_datasets["test"]["source"][idx])[:512])
                summary = summarizer(truncated_sentence)[0]["summary_text"]
                print(f""">>> Text: {truncated_sentence}""")

            else:
                summary = summarizer(processed_datasets["test"]["source"][idx])[0]["summary_text"]
                print(f""">>> Text: {processed_datasets["test"]["source"][idx]}""")

            print(f"""\n>>> Actual Summary: {processed_datasets["test"]["target"][idx]}""")
            print(f"\n>>> Predicted Summary: {summary}")

    print("\n\nTest sample index: ")
    index = input()
    while index != "q":
        print_summary(int(index))
        print("\n\nTest sample index: ")
        index = input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', required=True)
    args = parser.parse_args()
    main(args)
