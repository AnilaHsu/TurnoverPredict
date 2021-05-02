from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def load_xy(df: DataFrame):
    y = df.PerStatus.values
    X = df.drop(['PerStatus'], axis=1).values
    return X, y


def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=0)


# 無特徵篩選
def no_selection(df: DataFrame):
    X, y = load_xy(df)
    return X, y


# 描述統計特徵選擇
def descriptive_statistics_selection(df: DataFrame):
    X, y = load_xy(df)
    X = df[['管理層級', '專案時數', '專案總數', '生產總額', '榮譽數', '是否升遷', '升遷速度', '年齡層級', '年資層級A', '廠區代碼']].values
    return X, y


# 相關係數特徵選擇
def corr_selection(df: DataFrame):
    corr = df.corr().PerStatus.sort_values(ascending=False)
    corr = corr[corr > 0]
    print(corr.index)

    y = df.PerStatus.values
    X = df[corr.index]
    X.drop('PerStatus', axis=1, inplace=True)
    return X, y


# 過濾法_方差選擇法特徵選擇
def variance_selection(df: DataFrame):
    X, y = load_xy(df)
    var_selector = VarianceThreshold(threshold=0.1)
    X = var_selector.fit_transform(X)
    return X, y
