# Impact of Political Bias in Large Language Models on Downstream Task Performance

Carter Galbus (cgalbus@umich.edu), Haley Johnson (haleyej@umich.edu), Laura 
Kurek (lkurek@umich.edu), and Ankith Palakodati (apalakod@umich.edu)

This is our final project for EECS 592, Foundations of Artificial 
Intelligence, at the University of Michigan

Our project build's on Feng et al.'s paper ["From Pretraining Data to 
Language Models to Downstream Tasks: Tracking the Trails of Political Biases 
Leading to Unfair NLP Models"](https://arxiv.org/abs/2305.08283). We investigate if 
ensembling LLMs with different political biases improves downstream task 
performance on issues were there are clear partisan assymetries. In 
particular, we evaluate our political models on a climate denial detection 
dataset to see if partisan LLMs can learn to ignore political misinformation 
that appears in their training data or if the model with learn "bothside-ism" 

## Data
All datasets belong to their respective owners. Due to their large size, none of the language modeling corpuses are hosted on this repository. 

Language modeling:
* [POLITICS news corpus](https://aclanthology.org/2022.findings-naacl.101/)
* [Political subreddits](https://aclanthology.org/2021.eacl-main.152/)

Downstream tasks:
* [Twitter climate seniment](https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset)

## Repository Structure 
```
├── data                                    <- Data soruces used
│   └── climate_sentiment_test.csv          <- Climate denial training data 
│   └── climate_sentiment_train.csv         <- Climate denial testing data
|   └── twitter_sentiment_data.csv          <- Raw twitter sentiment data
| 
├── evaluation                              <- Evaluate LLMs
│   └── evaluate_mlm.py                     <- Analyze sentiment of responses
│   └── political_compass.jsonl             <- Statements from political compass test
│   └── run_eval.sh                         <- Script to run eval on all models
| 
├── models                                  <- Code to fine-tune LLMs
│   └── climate_denial_downstream.py        <- Run downstream tasks
│   └── fine_tune_political_models.py       <- Induce bias in LLMs
|   └── run.sh                              <- Script to fine-tune all models
│
├── src                                     <- Utility code
│   └── split_data.py                       <- Clean & split climate data  
|
└── README.md
```