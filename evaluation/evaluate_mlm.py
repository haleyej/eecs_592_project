import os
import json
import argparse
import pandas as pd
import eng_spacysentiment

from transformers import pipeline, RobertaTokenizerFast, AutoModelForMaskedLM


def run_eval(model_path:str, 
             output_file:str,
             eval_statements:list[str], 
             top_k:int = 15) -> None:
    
    '''
    Loads pretrained model, does masked language modeling 
    inference for a list of predefined sentences. Gets the top_k 
    masked tokens for each sentence 

    Saves results to output_file

    ARGUMENTS:
        model_path: path to pretrained model 
        output_file: path to save results to 
        eval_statements: list of statements to evaluate model with
        top_k: number of tokens to get for each inference 
    
    RETURNS:
        None
    '''
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)
    model =  AutoModelForMaskedLM.from_pretrained(model_path)
    nlp = eng_spacysentiment.load()

    mask_fill = pipeline('fill-mask', model = model, tokenizer = tokenizer, top_k = top_k)

    sentences_data = []
    for statement in eval_statements: 
        prompt = statement + ' I <mask> with this statement'
        top_k_words = mask_fill(prompt)
        sentence_data = []

        for i, word in enumerate(top_k_words):
            token = word.get('token_str', '').strip()
            sequence_sentiment = nlp(f"I {token} with this statement").cats

            sentence_data.append([prompt, 
                                  (i + 1), 
                                  token, 
                                  word.get('score', 0), 
                                  sequence_sentiment.get('positive'), 
                                  sequence_sentiment.get('negative'), 
                                  sequence_sentiment.get('neutral')])
        
        sentences_data.extend(sentence_data)

    df = pd.DataFrame(sentences_data)
    df.columns = ['sentence', 'position', 'token', 'score', 'positive_sent', 'negative_sent', 'neutral_sent']
    df.to_csv(output_file, index = False)



def load_political_statements(path:str) -> list[str]:
    '''
    loads in jsonlines file with data from the 
    political compass test

    ARGUMENTS:
        path: path to jsonl file with political compass statements 
              from https://github.com/BunsenFeng/PoliLean
    
    RETURNS:
        a list with each political compass statement
    '''
    statements = []
    with open(path) as f: 
        lines = json.load(f)
        for line in lines:
            statements.append(line.get('statement', ''))

    return statements


def main(args):
    eval_statements = load_political_statements(args['eval_path'])
    run_eval(args['model_path'], 
             args['output_file'], 
             eval_statements,
             args['top_k']) 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--top_k', type=int, default=15)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)