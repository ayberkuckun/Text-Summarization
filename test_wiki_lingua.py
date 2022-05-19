import argparse

import numpy as np
from transformers import pipeline
from datasets import load_dataset


def main(args):
    def process(examples):
        documents = [article["document"][-1] for article in examples['article'] if article["document"]]
        summaries = [article["summary"][-1] for article in examples['article'] if article["document"]]

        output = {}
        output["document"] = documents
        output["summary"] = summaries

        return output

    dataset_name = "wiki_lingua"
    dataset = load_dataset(dataset_name, "english")
    processed_datasets = dataset.map(process, batched=True, remove_columns=dataset["train"].column_names)
    processed_datasets = processed_datasets["train"].train_test_split(test_size=0.2, seed=42)

    model_checkpoint = args.model_checkpoint
    # model_checkpoint = f"results/{dataset_name}/checkpoint-5905"
    summarizer = pipeline("summarization", model=model_checkpoint)

    def print_summary(idx):
        summary = summarizer(processed_datasets["test"]["document"][idx])[0]["summary_text"]

        print(f""">>> Text: {processed_datasets["test"]["document"][idx]}""")
        print(f"""\n>>> Actual Summary: {processed_datasets["test"]["summary"][idx]}""")
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
