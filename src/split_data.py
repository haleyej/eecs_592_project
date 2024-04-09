import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split


def split_climate_data(path:str,
                       train_path:str = 'climate_sentiment_train.csv', 
                       test_path:str = 'climate_sentiment_test.csv', 
                       test_size:int = 0.4) -> None:
    
    '''
    simple helper function to split data
    '''

    df = pd.read_csv(path)

    # first class starts at 0
    df['sentiment'] = df['sentiment'] + 1

    train_df, test_df = train_test_split(df, test_size = test_size, random_state = 42)

    train_df.to_csv(train_path, index = False)
    test_df.to_csv(test_path, index = False)


def main():
    path = '../twitter_sentiment_data.csv'
    train_path = '../data/climate_sentiment_train.csv'
    test_path = '../data/climate_sentiment_test.csv'

    split_climate_data(path, train_path, test_path)

if __name__ == '__main__':
    main()