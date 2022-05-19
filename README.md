# Text-Summarization

## For client
To install dependencies: `npm install`

To run: `npm start`

## For server
Unfortunately there is currently no list of requirements as far as dependencies goes. 

It also requires to download and unpack the Amazon Fine Food dataset Reviews.csv file into the server/datasets folder. Furthermore, pre-trained, but non-finetuned models can be loaded by setting the flag LOCAL to False in util.py. Fine-tuned models were too large for storage here but can be provided upon request.

If dependencies are satisfied, to run: `python server.py`