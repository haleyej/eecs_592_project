## Impact of Political Bias in Large Language Models on Downstream Task Performance

Carter Galbus (cgalbus@umich.edu), Haley Johnson (haleyej@umich.edu), Laura 
Kurek (lkurek@umich.edu), and Ankith Palakodati (apalakod@umich.edu)

This is our final project for EECS 592, Foundations of Artificial 
Intelligence, at the University of Michigan

Our project build's on Sheng et al.'s paper [From Pretraining Data to Language 
Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to 
Unfair NLP Models](https://arxiv.org/abs/2305.08283). We investigate if 
ensembling LLMs with different political biases improves downstream task 
performance on issues were there are clear partisan assymetries. In 
particular, we evaluate our political models on a climate denial detection 
dataset to see if partisan LLMs can learn to ignore political misinformation 
that appears in their training data or if the model with learn "bothside-ism" 
