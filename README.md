# Text-Summarization

![Results](results.png?raw=true "Results")

![Outputs](outputs.png?raw=true "Outputs")

## For client
To install dependencies: `npm install`

To run: `npm start`

## For server
Unfortunately there is currently no list of requirements as far as dependencies goes. 

It also requires to download and unpack the Amazon Fine Food dataset Reviews.csv file into the server/datasets folder. Furthermore, pre-trained, but non-finetuned models can be loaded by setting the flag LOCAL to False in util.py. Fine-tuned models were too large for storage here but can be provided upon request.

If dependencies are satisfied, to run: `python server.py`

## For Trainings other than Amazon Fine Food

-- Trainings are named after their datasets.

- Datasets are downloaded automatically.

- Only for "sci-tldr.py" and "samsum.py" you need to provide an argument.

- For others just use "python "dataset".py"

- For "sci-tldr.py" and "samsum.py" provide "--model_checkpoint" argument.

- "--model_checkpoint" argument can be "facebook/bart-base" and "t5-small" for "samsum" and additionally "bert2gpt2" for "sci-tldr".

-- For evaluations use "test_"dataset".py" files for the respective datasets.

- You can run them with "--model_checkpoint" argument and input the test sample's index when asked.

- Fine-tuned models for testing can be provided upon request.
