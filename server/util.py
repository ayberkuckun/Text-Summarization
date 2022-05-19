import json
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import rouge
from datasets import load_dataset, load_metric, DatasetDict, DatasetBuilder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

#DATASETS = {"amazon_fine_food": "Reviews.csv"}
# Collection of loaded datasets
LOCAL = True
LOADED_DATASETS = {}
# Model configuration: name, checkpoint for model and tokenizer, reference to model and tokenizer, dataset of loaded datasets to use, column names for text and summary in dataset
MODEL_MAP = ["facebookBart_amazon_food", "facebookBart_samsum"] # 
MODELS = {
  "facebookBart_amazon_food": {"checkpoint": "model_facebookbart_food", "checkpoint-tokenizer": "facebook/bart-base", "checkpoint-online": "facebook/bart-base", "checkpoint-tokenizer-online": "facebook/bart-base", "model": None, "tokenizer": None, "dataset": "amazon_fine_food", "datasetIndex": None, "text": "Text", "summary": "Summary"},
  "facebookBart_samsum": {"checkpoint": "model_facebookbart_samsum", "checkpoint-tokenizer": "facebook/bart-base", "checkpoint-online": "facebook/bart-base", "checkpoint-tokenizer-online": "facebook/bart-base", "model": None, "tokenizer": None, "dataset": "samsum", "datasetIndex": None, "text": "dialogue", "summary": "summary"},
  "facebookBart_billsum": {"checkpoint": "model_facebookbart_billsum", "checkpoint-tokenizer": "facebook/bart-base", "checkpoint-online": "facebook/bart-base", "checkpoint-tokenizer-online": "facebook/bart-base", "model": None, "tokenizer": None, "dataset": "billsum", "datasetIndex": None, "text": "text", "summary": "summary"},
  "facebookBart_scitldr": {"checkpoint": "model_facebookbart_scitldr", "checkpoint-tokenizer": "facebook/bart-base", "checkpoint-online": "facebook/bart-base", "checkpoint-tokenizer-online": "facebook/bart-base", "model": None, "tokenizer": None, "dataset": "scitldr", "datasetIndex": 0, "text": "source", "summary": "target"},
  "facebookBart_wikilingua": {"checkpoint": "model_facebookbart_wiki_lingua", "checkpoint-tokenizer": "facebook/bart-base", "checkpoint-online": "facebook/bart-base", "checkpoint-tokenizer-online": "facebook/bart-base", "model": None, "tokenizer": None, "dataset": "wikilingua", "datasetIndex": -1, "text": "document", "summary": "summary"},
  "T5_samsum": {"checkpoint": "model_t5_samsum", "checkpoint-tokenizer": "t5-small", "checkpoint-online": "t5-small", "checkpoint-tokenizer-online": "t5-small", "model": None, "tokenizer": None, "dataset": "samsum", "datasetIndex": None, "text": "dialogue", "summary": "summary"},
  "T5_scitldr": {"checkpoint": "model_t5_scitldr_aic", "checkpoint-tokenizer": "t5-small", "checkpoint-online": "t5-small", "checkpoint-tokenizer-online": "t5-small", "model": None, "tokenizer": None, "dataset": "scitldr", "datasetIndex": 0, "text": "source", "summary": "target"},
  "BERT2GPT2_scitldr": {"checkpoint": "model_bert2gpt2_scitldr", "checkpoint-tokenizer": "bert2gpt2", "checkpoint-online": "bert2gpt2", "checkpoint-tokenizer-online": "bert2gpt2", "model": None, "tokenizer": None, "dataset": "scitldr", "datasetIndex": 0, "text": "source", "summary": "target"}
}

def decode(data):
  return json.loads(data)

def store_datasets(dataset, name, remove_cols=[]):
  if len(remove_cols) > 0:
    dataset = dataset.remove_columns(remove_cols)

  LOADED_DATASETS[name] = dataset

def split_store_datasets(dataset, name, remove_cols=[], test_size=0.1, no_validation=False, store=True, seed=42):
  print("split", name, type(dataset))
  if seed is not None:
    train_testvalid = dataset.train_test_split(test_size=test_size, seed=seed)
  else:
    train_testvalid = dataset.train_test_split(test_size=test_size)

  if no_validation:
    train_test_valid_dataset = DatasetDict({
      #'train': train_testvalid['train'],
      'test': train_testvalid['test']})
  else:
    valid_test = train_testvalid["test"].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
        #'train': train_testvalid['train'],
        #'valid': valid_test['train'],
        'test': valid_test['test']})
  if len(remove_cols) > 0:
    train_test_valid_dataset = train_test_valid_dataset.remove_columns(remove_cols)

  if store:
    LOADED_DATASETS[name] = train_test_valid_dataset
  else:
    return train_test_valid_dataset

def preprocess_wiki(examples):
    documents = [article["document"][-1] for article in examples['article'] if article["document"]]
    summaries = [article["summary"][-1] for article in examples['article'] if article["document"]]

    model_inputs = tokenizer(documents, max_length=512, truncation=True)
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summaries, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_datasets():
  print("LOADING DATASETS...", "(1/3)")
  dataset_food = load_dataset("csv", data_files="./datasets/Reviews.csv", split="train")
  dataset_samsum = load_dataset("samsum")
  dataset_billsum = load_dataset("billsum")
  dataset_reddit = load_dataset("reddit_tifu", "long", split="train")
  dataset_scitldr_aic = load_dataset("scitldr", "AIC")
  dataset_wikilingua = load_dataset("wiki_lingua", split="train")
  remove_cols = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time"]
  split_store_datasets(dataset_food, "amazon_fine_food", remove_cols)
  store_datasets(dataset_samsum, "samsum", ["id"])
  store_datasets(dataset_scitldr_aic, "scitldr", ["rouge_scores", "paper_id", "ic", "source_labels"])
  store_datasets(dataset_billsum, "billsum", ["title"])
  remove_cols = ["ups", "num_comments", "upvote_ratio", "score", "title"]
  split_store_datasets(dataset_reddit, "reddit", remove_cols, 0.2, True)

def generate_random_summary(m, dataset="amazon_fine_food"):
  results = []
  txt = "Text"
  summy = "Summary"
  ind = None

  if m is not None and MODELS[m]["datasetIndex"] is not None:
    ind = MODELS[m]["datasetIndex"]

  if m is not None and MODELS[m]["dataset"] is not None:
    dataset = MODELS[m]["dataset"]
    txt = MODELS[m]["text"]
    summy = MODELS[m]["summary"]

  rand_index = np.random.randint(LOADED_DATASETS[dataset].num_rows["test"])

  text = LOADED_DATASETS[dataset]["test"][rand_index][txt]
  truth = LOADED_DATASETS[dataset]["test"][rand_index][summy]

  if ind is not None:
    text = text[ind]
    truth = truth[ind]

  if m is not None:
    model = MODELS[m]["model"]
    tokenizer = MODELS[m]["tokenizer"]
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    model.to("cpu")
    inputs.to("cpu")
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results.append({"model": m, "summary": decoded_output})
  else:
    for m in MODELS.keys():
      model = MODELS[m]["model"]
      tokenizer = MODELS[m]["tokenizer"]
      inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
      model.to("cpu")
      inputs.to("cpu")
      outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
      decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

      results.append({"model": m, "summary": decoded_output})

  return {"originalText": text, "originalSummary": truth, "results": results}

def generate_summary(text, m):
  #m = m[0]
  model = MODELS[m]["model"]
  tokenizer = MODELS[m]["tokenizer"]
  inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
  model.to("cpu")
  inputs.to("cpu")
  outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True) # 
  decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

  return decoded_output

def load_models():
  print("LOADING MODELS...", "(2/3)")
  for m in MODELS:
    if m not in MODEL_MAP:
      continue
    print(m)

    if LOCAL:
      tokenizer = AutoTokenizer.from_pretrained(MODELS[m]["checkpoint-tokenizer"])
      model = AutoModelForSeq2SeqLM.from_pretrained(MODELS[m]["checkpoint"])
    else:
      tokenizer = AutoTokenizer.from_pretrained(MODELS[m]["checkpoint-tokenizer-online"])
      model = AutoModelForSeq2SeqLM.from_pretrained(MODELS[m]["checkpoint-online"])

    MODELS[m]["model"] = model
    MODELS[m]["tokenizer"] = tokenizer

def calculate_scores(text, summarized_text):
  if text is None:
    return [{"rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "rouge-w": 0}]
  apply_avg = True
  apply_best = True
  evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=2, limit_length=True, length_limit=100, length_limit_type='words', apply_avg=apply_avg, apply_best=apply_best, alpha=0.5, weight_factor=1.2, stemming=True) # Default F1_score alpha thingie
  
  all_hypothesis = [summarized_text]
  all_references = [text]

  scores = evaluator.get_scores(all_hypothesis, all_references)

  return [{"rouge-1": scores["rouge-1"]["f"], "rouge-2": scores["rouge-2"]["f"], "rouge-l": scores["rouge-l"]["f"], "rouge-w": scores["rouge-w"]["f"]}]

def lex_rank(text):
  num_sentences = 3
  parser = PlaintextParser.from_string(text, Tokenizer("english"))
  summarizer = LexRankSummarizer()
  summary = summarizer(parser.document, num_sentences)
  result = []

  for sentence in summary:
    result.append(str(sentence))
  
  return result

def latent_semantic_analysis(text):
  num_sentences = 3
  parser=PlaintextParser.from_string(text,Tokenizer("english"))
  summarizer = LsaSummarizer()
  summary = summarizer(parser.document, num_sentences)
  result = []

  for sentence in summary:
    result.append(str(sentence))
  
  return result

def every_other_word(text):
  text_arr = text.split(" ")
  result = []

  for i in range(0, len(text_arr), 2):
    result.append(text_arr[i])

  return result

# Mappings for functions, return result type of those functions, and arguments for those functions, for user input to be summarized and evaluated.
METHODS = [every_other_word, lex_rank, latent_semantic_analysis, generate_summary, generate_summary]
ARRAY_RES = [True, True, True, False, False]
ARGS = [False, False, False] + MODEL_MAP

def summarize(text, summary=None, method_num=2, arr_res=True):
  result = None
  if ARGS[method_num] is False:
    result = METHODS[method_num](text)
  else:

    result = METHODS[method_num](text, ARGS[method_num])

  if ARRAY_RES[method_num]:
    result = " ".join(result)
  
  scores = calculate_scores(summary, result)

  return result, scores