import argparse

import numpy as np
from transformers import pipeline
from datasets import load_dataset


def main(args):
    dataset_name = "samsum"
    dataset = load_dataset(dataset_name)

    model_checkpoint = args.model_checkpoint
    # model_checkpoint = f"results/{dataset_name}/checkpoint-4860"
    summarizer = pipeline("summarization", model=model_checkpoint)

    def print_summary(idx):
        summary = summarizer(dataset["test"]["dialogue"][idx])[0]["summary_text"]

        print(f""">>> Text: {dataset["test"]["dialogue"][idx]}""")
        print(f"""\n>>> Actual Summary: {dataset["test"]["summary"][idx]}""")
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
