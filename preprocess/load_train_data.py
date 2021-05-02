import pandas as pd

def load_train_data():
    df = pd.read_csv('data/train.csv')
    return df