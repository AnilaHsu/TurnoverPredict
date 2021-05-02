from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import fbeta_score


def train(x_train,x_test, y_train,y_test):

    tree_model = DecisionTreeClassifier(max_depth=35,criterion='entropy')
    tree_model.fit(x_train,y_train)
    y_pred = tree_model.predict(x_test)

    print('DecisionTree')
    print('訓練集準確率: ', tree_model.score(x_train, y_train))
    print('测試集準確率: ', tree_model.score(x_test, y_test))
    print(metrics.classification_report(y_test, y_pred))
    print('F beta score:',fbeta_score(y_test, y_pred, beta=1.5))
    return tree_model