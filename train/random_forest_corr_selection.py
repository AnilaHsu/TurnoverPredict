from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import fbeta_score


def train(x_train,x_test, y_train,y_test):
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    # print("Accuracy:",accuracy_score(y_test, y_pred))

    print('Random Forest')
    print('訓練集準確率: ', clf.score(x_train, y_train))
    print('测試集準確率: ', clf.score(x_test, y_test))
    print(metrics.classification_report(y_test, y_pred))
    print('F beta score:',fbeta_score(y_test, y_pred, beta=1.5))
    return clf
