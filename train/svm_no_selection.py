from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import fbeta_score


def train(x_train, x_test, y_train, y_test):

    svm = SVC()
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)

    print('SVM')
    print('訓練集準確率: ', svm.score(x_train, y_train))
    print('测試集準確率: ', svm.score(x_test, y_test))
    print(metrics.classification_report(y_test, y_pred))
    print('F beta score:', fbeta_score(y_test, y_pred, beta=1.5))
    return svm
