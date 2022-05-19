import http.server
import socketserver
import json
from util import summarize, decode, load_datasets, load_models, generate_random_summary, calculate_scores, LOADED_DATASETS, MODELS, MODEL_MAP

SUMMARIZATION_METHODS = ["every_other", "lex_rank", "latent_semantic_analysis"] + [m + "_finetune" for m in MODEL_MAP]

class HTTPHandler(http.server.BaseHTTPRequestHandler):
  def do_GET(self):
    print("GET", self.path)

    self.send_response(200)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("content-type", "application-json")
    self.end_headers()

    datasets = []

    #for dataset in LOADED_DATASETS.keys():
    for m in MODEL_MAP:
      print(m, MODELS[m].keys())
      datasets.append(MODELS[m]["dataset"])

    data = json.dumps({"methods": SUMMARIZATION_METHODS, "datasets": datasets})
    self.wfile.write(data.encode(encoding='utf_8'))

  def do_POST(self):
    print("POST", self.path)

    body = self.rfile.read(int(self.headers['Content-Length']))
    data = decode(body)["data"]

    if self.path == "/":
      method_num = data["method"]
      text = data["text"]
      summary = data["summary"]

      if len(summary) == 0:
        summary = None
      
      summarized_text, scores = summarize(text, summary, method_num)

      self.send_response(200)
      self.send_header("Access-Control-Allow-Origin", "*")
      self.send_header("content-type", "application-json")
      self.end_headers()

      data = json.dumps({"output": summarized_text, "metrics" : scores})
      self.wfile.write(data.encode(encoding='utf_8'))
    elif self.path == "/generate":
      model_num = data["dataset_num"]
      model = MODEL_MAP[model_num]

      self.send_response(200)
      self.send_header("Access-Control-Allow-Origin", "*")
      self.send_header("content-type", "application-json")
      self.end_headers()

      summary = generate_random_summary(model)
      
      originalSummary = summary["originalSummary"]
      generatedSummary = summary["results"][0]["summary"]

      if len(originalSummary) == 0:
        originalSummary = None

      scores = calculate_scores(originalSummary, generatedSummary)

      data = json.dumps({"output": summary, "metrics": scores})

      self.wfile.write(data.encode(encoding='utf_8'))

  def do_OPTIONS(self):
    print("OPTIONS")

    self.send_response(200)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "*")
    self.send_header("Access-Control-Allow-Headers", "*")
    self.end_headers()

Handler = HTTPHandler
port = 1234

with http.server.HTTPServer(("", port), Handler) as httpd:
  load_datasets()
  load_models()
  print("Listening on", port, "(3/3)")
  httpd.serve_forever()