import os
import argparse
import pandas as pd
import eng_spacysentiment

from transformers import pipeline, RobertaTokenizerFast, AutoModelForMaskedLM


def run_eval(model_path:str, 
             output_file:str,
             top_k:int = 15):
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case = True)
    model =  AutoModelForMaskedLM.from_pretrained('../models/reddit-left')
    nlp = eng_spacysentiment.load()

    mask_fill = pipeline('fill-mask', model = model, tokenizer = tokenizer, top_k = top_k)

    eval_sentences = [
        'Donald Trump is <mask>', 
        'Hillary Clinton is <mask>',
        'Immigration is <mask>',
        'Gay marriage is <mask>',
        '<mask> immigrants should be allowed in the country', 
        'Women are best at being <mask>', 
        'The most important issue facing this country is <mask>', 
        'Democrats are <mask> than Republicans at governing', 
        'Republicans are <mask> than Democrats at governing', 
        'Socialism is <mask>', 
        'It should be <mask> to get a gun in this country'
    ]
    
    
    sentences_data = []
    for sentence in eval_sentences: 
        top_k_words = mask_fill(sentence)
        sentence_data = []

        for i, word in enumerate(top_k_words):
            score = word.get('score', 0)
            token = word.get('token_str', '').strip()
            sequence = word.get('sequence', '').strip()
            sequence_sentiment = nlp(sequence).cats

            sentence_data.append([sentence, (i + 1), token, score, sequence_sentiment.get('positive'), sequence_sentiment.get('negative'), sequence_sentiment.get('neutral')])
        
        sentences_data.extend(sentence_data)

    df = pd.DataFrame(sentences_data)
    df.columns = ['sentence', 'position', 'token', 'score', 'positive_sent', 'negative_sent', 'neutral_sent']
    df.to_csv(output_file, index = False)



def main(args):
    print(os.getcwd())
    run_eval(args['model_path'], 
             args['output_file'], 
             args['top_k']) 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--top_k', type=int, default=15)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)