import os 
import torch
import evaluate
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm
#eecs592s001w24_class

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, RobertaTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer

metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def process_climate_data(path:str) -> tuple:
    '''
    helper function, loads climate data

    ARGUMENTS:
        path: path to csv file with climate data
    
    RETURNS:
        a tuple with two elements
            1) the texts
            2) the labels
    '''
    df = pd.read_csv(path)
    texts = df['message'].to_list()
    labels = df['sentiment'].to_list()
    return texts, labels 


def compute_metrics(pred) -> dict:
    '''
    helper function for training step, computes 
    evaluate metrics 

    ARGUMENTS: 
        pred: predictions on the evaluation set from the model 

    RETURNS:
        dictionary mapping metrics to values
    '''
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)


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
                            eval_steps:int = 1000,
                            learning_rate = 1e-5, 
                            epochs:int = 5, 
                            weight_decay:float = 0.01) -> None:
    
    '''
    classification pretraining 
    sets up data, model and training arguments

    saves model results to the specified output directory
    '''
    
    texts, labels = process_climate_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size = eval_size)
    num_classes = len(set(labels))

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels = num_classes).to(device)

    climate_dataset = ClimateClassificationDataset(X_train, y_train, tokenizer, max_len)
    eval_dataset = ClimateClassificationDataset(X_val, y_val, tokenizer, max_len)

    data_collator = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'max_length')

    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        evaluation_strategy = "steps",
        eval_steps = eval_steps,
        logging_dir = logging_dir,
        logging_strategy = "steps",
        logging_steps = 50,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        warmup_steps = 500,
        save_strategy = "steps",
        save_steps = eval_steps,
        load_best_model_at_end = True)

    trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = climate_dataset,
            eval_dataset = eval_dataset,
            data_collator = data_collator,
            tokenizer = tokenizer, 
            compute_metrics = compute_metrics)

    # Use tqdm to create a progress bar
    with tqdm(total=training_args.num_train_epochs, desc="Training") as pbar:
        trainer.train()
        pbar.update(1)

    torch.save(model.state_dict(), f"{model_name}-classification-weights.pth")


def main(args):
    # how to run from command line 
    # python3 climate_denial_downstream.py --pretrained_model_path='reddit-left' --model_name='reddit-left' --data_path='../data/climate_sentiment_test.csv' 
    # replace paths with the correct ones for your dir structure
    # should also probably specify logging / output door so your life is not miserable 
    if args['mode'] == 'train':
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
                                args['eval_steps'],
                                args['learning_rate'],
                                args['epochs'], 
                                args['weight_decay'])
    elif args['mode'] == 'test':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForSequenceClassification.from_pretrained(args['pretrained_model_path']).to(device)
        model.load_state_dict(torch.load(f"{args['model_name']}-classification-weights.pth"))
        model.eval()

        test_texts, test_labels = process_climate_data(args['test_data_path'])
        test_dataset = ClimateClassificationDataset(test_texts, test_labels, tokenizer, 512)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'max_length'))

        all_predictions = []
        all_labels = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = torch.argmax(outputs.logits, dim=1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(batch['labels'])
            
        all_predictions = np.array(all_predictions)
        all_labels = torch.stack(all_labels).numpy()
        scores = metrics.compute(predictions = all_predictions, references = all_labels)
        print(f'{args['model_name']}:')
        print(scores)
    elif args['mode'] == 'test-ensemble':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        redditCenterModel = AutoModelForSequenceClassification.from_pretrained('/home/apalakod/eecs_592_project/zip_gg_files/fine-tuned_models/reddit-center').to(device)
        redditCenterModel.load_state_dict(torch.load(f"reddit-center-classification-weights.pth"))
        redditCenterModel.eval()

        redditLeftModel = AutoModelForSequenceClassification.from_pretrained('/home/apalakod/eecs_592_project/zip_gg_files/fine-tuned_models/reddit-left').to(device)
        redditLeftModel.load_state_dict(torch.load(f"reddit-left-classification-weights.pth"))
        redditLeftModel.eval()

        redditRightModel = AutoModelForSequenceClassification.from_pretrained('/home/apalakod/eecs_592_project/zip_gg_files/fine-tuned_models/reddit-right').to(device)
        redditRightModel.load_state_dict(torch.load(f"reddit-right-classification-weights.pth"))
        redditRightModel.eval()

        newsCenterModel = AutoModelForSequenceClassification.from_pretrained('/home/apalakod/eecs_592_project/zip_gg_files/fine-tuned_models/news-center').to(device)
        newsCenterModel.load_state_dict(torch.load(f"news-center-classification-weights.pth"))
        newsCenterModel.eval()

        newsLeftModel = AutoModelForSequenceClassification.from_pretrained('/home/apalakod/eecs_592_project/zip_gg_files/fine-tuned_models/news-left').to(device)
        newsLeftModel.load_state_dict(torch.load(f"news-left-classification-weights.pth"))
        newsLeftModel.eval()

        newsRightModel = AutoModelForSequenceClassification.from_pretrained('/home/apalakod/eecs_592_project/zip_gg_files/fine-tuned_models/news-right').to(device)
        newsRightModel.load_state_dict(torch.load(f"news-right-classification-weights.pth"))
        newsRightModel.eval()

        test_texts, test_labels = process_climate_data(args['test_data_path'])
        test_dataset = ClimateClassificationDataset(test_texts, test_labels, tokenizer, 512)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'max_length'))

        reddit_predictions = []
        news_predictions = []
        all_labels = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                rlOut = redditLeftModel(input_ids=input_ids, attention_mask=attention_mask)
                rrOut = redditRightModel(input_ids=input_ids, attention_mask=attention_mask)
                rcOut = redditCenterModel(input_ids=input_ids, attention_mask=attention_mask)
                nlOut = newsLeftModel(input_ids=input_ids, attention_mask=attention_mask)
                nrOut = newsRightModel(input_ids=input_ids, attention_mask=attention_mask)
                ncOut = newsCenterModel(input_ids=input_ids, attention_mask=attention_mask)

            reddit_predictions.extend(torch.argmax(rlOut.logits+rrOut.logits+rcOut.logits, dim=1).tolist())
            news_predictions.extend(torch.argmax(nlOut.logits+nrOut.logits+ncOut.logits, dim=1).tolist())
            all_labels.extend(batch['labels'])
            
        news_predictions = np.array(news_predictions)
        reddit_predictions = np.array(reddit_predictions)
        all_labels = torch.stack(all_labels).numpy()
        reddit_scores = metrics.compute(predictions = reddit_predictions, references = all_labels)
        news_scores = metrics.compute(predictions = news_predictions, references = all_labels)
        print(f'reddit-ensemble-scores:')
        print(reddit_scores)
        print(f'news-ensemble-scores:')
        print(news_scores)
    elif args['mode'] == 'test-base':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        classification_pretrain(tokenizer,
                                'distilroberta-base',
                                args['data_path'],
                                args['model_name'],
                                args['output_dir'], 
                                args['logging_dir'], 
                                args['eval_size'], 
                                args['max_len'],
                                args['batch_size'],
                                args['eval_steps'],
                                args['learning_rate'],
                                args['epochs'], 
                                args['weight_decay'])
        
        bertModel = AutoModelForSequenceClassification.from_pretrained('distilroberta-base').to(device)
        bertModel.load_state_dict(torch.load(f"base-classification-weights.pth"))
        bertModel.eval()

        test_texts, test_labels = process_climate_data(args['test_data_path'])
        test_dataset = ClimateClassificationDataset(test_texts, test_labels, tokenizer, 512)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'max_length'))

        all_predictions = []
        all_labels = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                out = bertModel(input_ids=input_ids, attention_mask=attention_mask)

            all_predictions.extend(torch.argmax(out.logits, dim=1).tolist())
            all_labels.extend(batch['labels'])
            
        all_predictions = np.array(all_predictions)
        all_labels = torch.stack(all_labels).numpy()
        base_scores = metrics.compute(predictions = all_predictions, references = all_labels)
        print(f'base-scores:')
        print(base_scores)
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str, default=os.getcwd())
    parser.add_argument('--logging_dir', type=str, default=os.getcwd()),
    parser.add_argument('--eval_size', type=int, default=0.2)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--mode', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)
