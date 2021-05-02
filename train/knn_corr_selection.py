from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import fbeta_score


def train(x_train,x_test, y_train,y_test):

    knn = KNeighborsClassifier()
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    print('GaussianNB')
    print('訓練集準確率: ', knn.score(x_train, y_train))
    print('测試集準確率: ', knn.score(x_test, y_test))
    print('测試集召回率:', metrics.classification_report(y_test,y_pred))
    print('F beta score:',fbeta_score(y_test, y_pred, beta=1.5))
    return knn