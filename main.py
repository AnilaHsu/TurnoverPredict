from preprocess.load_train_data import load_train_data
from preprocess.data_preprocess import remove_empty_data, standard_scaler
from preprocess.data_feature_selection import split_data, no_selection, descriptive_statistics_selection, \
    corr_selection, variance_selection
from train import knn_no_selection, gnb_no_selection, svm_no_selection, decision_no_selection, \
    random_forest_no_selection, knn_descriptive_statistics_selection, gnb_descriptive_statistics_selection, \
    decision_descriptive_statistics_selection, random_forest_descriptive_statistics_selection, knn_corr_selection, \
    gnb_corr_selection, decision_corr_selection, random_forest_corr_selection, knn_variance_selection, \
    decision_variance_selection, random_forest_variance_selection, gnb_variance_selection
from pandas import DataFrame

from visualization.data_visualization import draw_roc


def run_no_selection(df: DataFrame):
    X, y = no_selection(df)
    x_train, x_test, y_train, y_test = split_data(X, y)
    x_train, x_test = standard_scaler(x_train, x_test)

    knn_model = knn_no_selection.train(x_train, x_test, y_train, y_test)
    gnb_model = gnb_no_selection.train(x_train, x_test, y_train, y_test)
    tree_model = decision_no_selection.train(x_train, x_test, y_train, y_test)
    random_forest_model = random_forest_no_selection.train(x_train, x_test, y_train, y_test)
    svm_model = svm_no_selection.train(x_train, x_test, y_train, y_test)

    knn_y_pred = knn_model.predict_proba(x_test)
    gnb_y_pred = gnb_model.predict_proba(x_test)
    tree_y_pred = tree_model.predict_proba(x_test)
    random_forest_y_pred = random_forest_model.predict_proba(x_test)

    draw_roc(y_test, knn_y_pred, gnb_y_pred, tree_y_pred, random_forest_y_pred)


def run_descriptive_statistics_selection(df: DataFrame):
    X, y = descriptive_statistics_selection(df)
    x_train, x_test, y_train, y_test = split_data(X, y)
    x_train, x_test = standard_scaler(x_train, x_test)

    knn_model = knn_descriptive_statistics_selection.train(x_train, x_test, y_train, y_test)
    gnb_model = gnb_descriptive_statistics_selection.train(x_train, x_test, y_train, y_test)
    tree_model = decision_descriptive_statistics_selection.train(x_train, x_test, y_train, y_test)
    random_forest_model = random_forest_descriptive_statistics_selection.train(x_train, x_test, y_train, y_test)

    knn_y_pred = knn_model.predict_proba(x_test)
    gnb_y_pred = gnb_model.predict_proba(x_test)
    tree_y_pred = tree_model.predict_proba(x_test)
    random_forest_y_pred = random_forest_model.predict_proba(x_test)

    draw_roc(y_test, knn_y_pred, gnb_y_pred, tree_y_pred, random_forest_y_pred)


def run_corr(df: DataFrame):
    X, y = corr_selection(df)
    x_train, x_test, y_train, y_test = split_data(X, y)
    x_train, x_test = standard_scaler(x_train, x_test)

    knn_model = knn_corr_selection.train(x_train, x_test, y_train, y_test)
    gnb_model = gnb_corr_selection.train(x_train, x_test, y_train, y_test)
    tree_model = decision_corr_selection.train(x_train, x_test, y_train, y_test)
    random_forest_model = random_forest_corr_selection.train(x_train, x_test, y_train, y_test)

    knn_y_pred = knn_model.predict_proba(x_test)
    gnb_y_pred = gnb_model.predict_proba(x_test)
    tree_y_pred = tree_model.predict_proba(x_test)
    random_forest_y_pred = random_forest_model.predict_proba(x_test)

    draw_roc(y_test, knn_y_pred, gnb_y_pred, tree_y_pred, random_forest_y_pred)


def run_variance(df: DataFrame):
    X, y = variance_selection(df)
    x_train, x_test, y_train, y_test = split_data(X, y)
    x_train, x_test = standard_scaler(x_train, x_test)

    knn_model = knn_variance_selection.train(x_train, x_test, y_train, y_test)
    gnb_model = gnb_variance_selection.train(x_train, x_test, y_train, y_test)
    tree_model = decision_variance_selection.train(x_train, x_test, y_train, y_test)
    random_forest_model = random_forest_variance_selection.train(x_train, x_test, y_train, y_test)

    knn_y_pred = knn_model.predict_proba(x_test)
    gnb_y_pred = gnb_model.predict_proba(x_test)
    tree_y_pred = tree_model.predict_proba(x_test)
    random_forest_y_pred = random_forest_model.predict_proba(x_test)

    draw_roc(y_test, knn_y_pred, gnb_y_pred, tree_y_pred, random_forest_y_pred)


if __name__ == "__main__":
    df = load_train_data()
    new_df = remove_empty_data(df)

    run_no_selection(new_df)
    run_descriptive_statistics_selection(new_df)
    run_corr(new_df)
    run_variance(new_df)
