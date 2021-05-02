import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

# Data visualization
# 離職人數占比
from sklearn.metrics import roc_curve, auc


def draw_perstatus_pie(df: DataFrame):
    data_left = df.PerStatus.value_counts()
    plt.pie([data_left[0] / df.PerStatus.count(), data_left[1] / df.PerStatus.count()], labels=['未離職', '已離職'],
            autopct='%1.1f%%',
            shadow=True, startangle=90)
    print('總人数:', df.PerStatus.count())
    print(data_left)


# Sex占比
def draw_sex_pie(df: DataFrame):
    data_sex = df.sex.value_counts()
    plt.pie([data_sex[0] / df.sex.count(), data_sex[1] / df.sex.count()], labels=['0', '1'], autopct='%1.1f%%',
            shadow=True, startangle=90)
    print('總人数:', df.sex.count())
    print(data_sex)


# 專案總數對離職影響
def draw_number_of_projects_boxplot(df: DataFrame):
    plt.rcParams['font.family'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    sns.boxplot(df['PerStatus'], df['專案總數'])


# 年齡層級對離職影響
def draw_age_level_boxplot(df: DataFrame):
    plt.rcParams['font.family'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    sns.boxplot(df['PerStatus'], df['年齡層級'])


# 年資層級Ａ對離職的影響
def draw_seniority_level_A_Histogram(df: DataFrame):
    table = pd.crosstab(df.年資層級Ａ, df.PerStatus)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.xlabel('PerStatus')
    plt.ylabel('年資層級Ａ')


# 廠區代碼對離職的影響
def draw_factory_code_Histogram(df: DataFrame):
    table = pd.crosstab(df.廠區代碼, df.PerStatus)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.xlabel('PerStatus')
    plt.ylabel('廠區代碼')


# ROC 曲線
def generate_knn_values(y_test, y_pred):
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(
        y_test, y_pred[:, 1])
    roc_auc_knn = auc(fpr_knn, tpr_knn)
    return fpr_knn, tpr_knn, roc_auc_knn


def generate_gnb_values(y_test, y_pred):
    fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(
        y_test, y_pred[:, 1])
    roc_auc_gnb = auc(fpr_gnb, tpr_gnb)
    return fpr_gnb, tpr_gnb, roc_auc_gnb


def generate_tree_values(y_test, y_pred):
    fpr_tree, tpr_tree, thresholds_tree = roc_curve(
        y_test, y_pred[:, 1])
    roc_auc_tree = auc(fpr_tree, tpr_tree)
    return fpr_tree, tpr_tree, roc_auc_tree


def generate_random_forest_values(y_test, y_pred):
    fpr_clf, tpr_clf, thresholds_clf = roc_curve(
        y_test, y_pred[:, 1])
    roc_auc_clf = auc(fpr_clf, tpr_clf)
    return fpr_clf, tpr_clf, roc_auc_clf


# 繪製 ROC 曲線
def draw_roc(y_test, knn_y_pred, gnb_y_pred, tree_y_pred, random_forest_y_pred):

    fpr_knn, tpr_knn, roc_auc_knn = generate_knn_values(y_test, knn_y_pred)
    fpr_gnb, tpr_gnb, roc_auc_gnb = generate_gnb_values(y_test, gnb_y_pred)
    fpr_tree, tpr_tree, roc_auc_tree = generate_tree_values(y_test, tree_y_pred)
    fpr_clf, tpr_clf, roc_auc_clf = generate_random_forest_values(y_test, random_forest_y_pred)

    plt.subplots(figsize=(7, 5.5))
    plt.plot(fpr_knn, tpr_knn, lw=2, label='Knn(area = %0.2f)' % roc_auc_knn)
    plt.plot(fpr_gnb, tpr_gnb, lw=2, label='GaussianNB(area = %0.2f)' % roc_auc_gnb)
    plt.plot(fpr_tree, tpr_tree, lw=2, label='DecisionTree(area = %0.2f)' % roc_auc_tree)
    plt.plot(fpr_clf, tpr_clf, lw=2, label='RandomForest(area = %0.2f)' % roc_auc_clf)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
