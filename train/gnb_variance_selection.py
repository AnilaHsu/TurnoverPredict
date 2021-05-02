from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import fbeta_score


def train(x_train,x_test, y_train,y_test):

    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred = gnb.predict(x_test)
    print('GaussianNB')
    print('訓練集準確率: ', gnb.score(x_train, y_train))
    print('测試集準確率: ', gnb.score(x_test, y_test))
    print('测試集召回率:', metrics.classification_report(y_test,y_pred))
    print('F beta score:',fbeta_score(y_test, y_pred, beta=1.5))
    return gnb