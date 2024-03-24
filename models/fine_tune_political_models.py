import os
import json
import wandb
import numpy as np 
import pandas as pd
from tqdm import tqdm
from typing import Literal
from sklearn.model_selection import train_test_split
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForMaskedLM

from torch.utils.data import DataLoader, Dataset



def process_big_news_dataset(label:Literal['center', 'left', 'right'], path:str) -> list[str]:
    '''
    load & process data from the BIGNEWS dataset 

    each folder in the BIGNEWS dataset contains 1+ 
    json files with text from news outlets with a particular political leaning 

    ARGUMENTS:
        label: what ideology you want to process data for 
               - should be 'center', 'left', or 'right'
        path: path to the folders

    RETURNS:
        list of strings containing the unprocessed text
    '''

    # labels from original paper: https://aclanthology.org/2022.findings-naacl.101.pdf
    # see figure 1
    label_to_files = {
        'right': ['train_fox.json', 'train_breitbart.json', 'train_wat.json',], 
        'center': ['train_ap.json', 'train_hill.json', 'train_usatoday.json'], 
        'left': ['train_wpo.json', 'train_cnn.json',  'train_nyt.json', 'train_dailykos.json', 'train_hpo.json']
    }

    target_files = label_to_files.get(label.lower(), [])

    texts = []
    folders = [folder for folder in os.listdir(path) is os.path.isdir(folder)]
    for folder in folders:
        folder_files = os.listdir(os.path.join(path, folder))
        for file in folder_files: 
            if file in target_files:
                with open(os.path.join(path, folder, file)) as f:
                    lines = json.loads(f.read())
                    for line in lines:
                        texts.append(" ".join(line.get('text', '')))
    return texts


def process_reddit_dataset(label:Literal['center', 'left', 'right'], path:str) -> list[str]:
    '''
    load & process data from the reddit dataset

    each file in the reddit dataset is a .txt file
    the file's name notes what kind of content is in it (e.g. reddit_left_post_trump.txt)


    ARGUMENTS:
        label: what ideology you want to process data for 
               - should be 'center', 'left', or 'right'
        path: path to the .txt files
    
    RETURNS:
        list of strings containing the unprocessed text
    '''
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    texts = []
    for file in files:
        if label in file:  
            with open(os.path.join(path, file)) as f:
                text = f.readlines()
                texts.extend(text)
    return texts



class MediaMLMDataset(Dataset):
    '''
    helper class for loading datasets
    '''
    def __init__(self, data:list[str], tokenizer, max_len:int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        
    
    def __getitem__(self, index):
        text = self.data[index]
        text = text.strip()
        
        return self.tokenizer(
                    text = text, 
                    padding = True, 
                    truncation = True, 
                    max_length = self.max_len, 
                    return_attention_mask = True, 
                    add_special_tokens = True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = False,
                    return_offsets_mapping = False)
        
    def __len__(self):
        return len(self.data)
        


def fine_tune_model(path:str, 
                    data_source:Literal['reddit', 'news'], 
                    tokenizer, 
                    output_dir:str, 
                    logging_dir:str, 
                    label:str = Literal['center', 'left', 'right'], 
                    eval_size:float = 0.7, 
                    max_len:int = 512, 
                    mlm_prob:float = 0.15,
                    batch_size:int = 8,
                    learning_rate:float = 5e-5,
                    epochs:int = 5,
                    weight_decay:float = 0.01) -> None:
    '''
    function to fine tune roberta base model on political corpus 

    fine tunes using masked language modeling

    ARGUMENTS:

        SET UP ARGUMENTS: 
            path: location of the data
                - if data_source == 'news' this should be the path to the folder 
                with the files from the BIGNEWS corpus 
                - e.g. if you have project/data/BIGNEWS/... and you want to train on the left leaning
                news sources, set the path to project/data/BIGNEWS
            data_source: what corpus you're pretraining on, either 'reddit' or 'news' 
            tokenizer: tokenizer to use on text
            output_dir: directory where saved model weights go 
                - should follow the format {data_source}-{label}-output
            logging_dir: directory where intermediate outputs go 
                - should follow the format {data_source}-{label}-logging
            label: what ideology you want to process data for 
                - should be 'center', 'left', or 'right'

        PREPROCESSING ARGUMENTS:
            eval_size: what percentage of the dataset to set away from evaluation during fine tuning 
                       - required, must be > 0
            max_len: maximum length of sequences
            mlm_prob: what percentage of the text gets masked during fine tuning         

        TRAINING ARGUMENTS: 
            batch_size: how many training examples to look at before updating weights
            learning_rate: how much the model updates 
            epochs: number of times the model looks at the data
            weight_decay: penalty amount on loss function, helps prevent overfitting

    RETURNS: 
        does not return anything, saves weights to output_dir and intermediate results to logging_dir
    '''

    if data_source.lower() == 'reddit':
        texts = process_reddit_dataset(label, path)
    elif data_source.lower() == 'news':
        texts = process_big_news_dataset(label, path)
    else:
        raise Exception('invalid data source')

    train_texts, eval_texts =  train_test_split(texts, test_size = eval_size) 

    mlm_dataset = MediaMLMDataset(train_texts, tokenizer, max_len)
    eval_dataset = MediaMLMDataset(eval_texts, tokenizer, max_len)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer, 
        mlm = True, 
        mlm_probability = mlm_prob)
    
    model_checkpoint = "roberta-base"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        evaluation_strategy = "epoch",
        logging_dir = logging_dir,
        logging_strategy = "steps",
        logging_steps = 10,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        warmup_steps = 500,
        save_strategy = "epoch",
        load_best_model_at_end = True,
        save_total_limit = 2)
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = mlm_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
        tokenizer = tokenizer,
    )

    trainer.train()

def train_all_models():
    pass 

def main():
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)

    fine_tune_model('../data/partisan_media', 'reddit')


if __name__ == "__main__":
    main()
