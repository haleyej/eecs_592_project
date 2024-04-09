import os 
import torch
import argparse
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, RobertaTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer


def process_climate_data(path:str) -> tuple:
    '''
    helper function, loads climate data
    '''
    df = pd.read_csv(path)
    texts = df['message'].to_list()
    labels = df['sentiment'].to_list()
    return texts, labels 



class ClimateClassificationDataset(Dataset):
    '''
    helper class for loading datasets
    '''
    def __init__(self, data:list[str], labels:list[str], tokenizer, max_len:int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index) -> tuple:
        text = self.data[index]
        label = self.labels[index]
        text = text.strip()
        
        output_dict = self.tokenizer(
                    text = text, 
                    padding = True, 
                    truncation = True, 
                    max_length = self.max_len, 
                    return_attention_mask = True, 
                    add_special_tokens = True,
                    return_special_tokens_mask = False,
                    return_token_type_ids = False,
                    return_offsets_mapping = False)
        
        output_dict['labels'] = [label]
        return output_dict
        
    def __len__(self) -> int:
        return len(self.data)
    

def classification_pretrain(tokenizer,
                            pretrained_model_path:str,  
                            data_path:str,
                            model_name:str,
                            output_dir:str, 
                            logging_dir:str,
                            eval_size:int = 0.2, 
                            max_len:int = 512,
                            batch_size:int = 8,
                            learning_rate = 1e-5, 
                            epochs:int = 5, 
                            weight_decay:float = 0.01) -> None:
    
    texts, labels = process_climate_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size = eval_size)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path)   

    climate_dataset = ClimateClassificationDataset(X_train, y_train, tokenizer, max_len)
    eval_dataset = ClimateClassificationDataset(X_val, y_val, tokenizer, max_len)

    data_collator = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'max_length')

    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        evaluation_strategy = "epoch",
        logging_dir = logging_dir,
        logging_strategy = "steps",
        logging_steps = 50,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        warmup_steps = 500,
        save_strategy = "epoch",
        load_best_model_at_end = True)
    
    trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = climate_dataset,
            eval_dataset = eval_dataset,
            data_collator = data_collator,
            tokenizer = tokenizer)

    trainer.train()
    torch.save(model.state_dict(), f"{model_name}-weights.pth")




def main(args):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)

    classification_pretrain(tokenizer,
                            args['pretrained_model_path'],
                            args['data_path'],
                            args['model_name'],
                            args['output_dir'], 
                            args['logging_dir'], 
                            args['eval_size'], 
                            args['max_len'],
                            args['batch_size'],
                            args['learning_rate'],
                            args['epochs'], 
                            args['weight_decay'])
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str, default='wafjkalshfkals')
    parser.add_argument('--output_dir', type=str, default=os.getcwd())
    parser.add_argument('--logging_dir', type=str, default=os.getcwd()),
    parser.add_argument('--eval_size', type=int, default=0.2)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)
