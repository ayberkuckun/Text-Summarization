import argparse

import numpy as np
from nltk import word_tokenize
from transformers import pipeline
from datasets import load_dataset


def main(args):
    dataset_name = "reddit_tifu"
    dataset = load_dataset(dataset_name, "long")
    ds = dataset["train"].train_test_split(test_size=0.2, seed=42)

    model_checkpoint = args.model_checkpoint
    # model_checkpoint = f"results/{dataset_name}/checkpoint-10530"
    summarizer = pipeline("summarization", model=model_checkpoint)

    def print_summary(idx):
        truncated_sentence = " ".join(word_tokenize(ds["test"]["documents"][idx])[:512])
        summary = summarizer(truncated_sentence)[0]["summary_text"]

        print(f""">>> Text: {ds["test"]["documents"][idx]}""")
        print(f"""\n>>> Actual Summary: {ds["test"]["tldr"][idx]}""")
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
