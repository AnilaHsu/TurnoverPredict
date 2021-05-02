import pandas as pd
from sklearn.preprocessing import StandardScaler

def remove_empty_data(df):
    new_df = df.copy()
    new_df.drop(['最高學歷', '畢業學校類別'], axis=1, inplace=True)
    new_df.dropna(inplace=True)
    return new_df


def standard_scaler(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    return x_train, x_test
