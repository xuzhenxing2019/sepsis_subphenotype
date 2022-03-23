"""

Prediction

"""
seed = 42
random_state= seed

import pandas as pd

import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import os
import sys
import pickle as pkl
import scipy
import shap
import math
import numpy as np, scipy.stats as st
# np.random.seed()
from sklearn.linear_model import LogisticRegression,ElasticNet
# from sklearn.svm import SVC
from xgboost import XGBClassifier,plot_importance

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.under_sampling import RandomUnderSampler,TomekLinks,NeighbourhoodCleaningRule,NearMiss,ClusterCentroids,EditedNearestNeighbours
from imblearn.pipeline import Pipeline

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve,auc,classification_report

import matplotlib.pyplot as plt
# from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold,KFold
from scipy import interp
# import category_encoders as ce
# from fancyimpute import KNN
# from fancyimpute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
import datetime

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

def build_classifier_0(X_train,y_train,X_test,y_test,feature_cols,classifier_flag, n_fold,file_path): #X[train], y[train],X[test]
# change label so as to the objective case is "1", the objective of change label is to keep  classifier.predict_proba(X_text)[:, 1]   "1" is case
    print('---------------------this is classifier_0----------------------')
    print('y_train_number_before_change_label',y_train.value_counts())
    # first replace "0" with 4, then replace 1,2,3, with "0", then replace 4 with "1"
    y_train = y_train.replace(0, 4)
    y_test = y_test.replace(0, 4)

    # then replace 1,2,3, with "0",
    y_train = y_train.replace(1, 0)
    y_train = y_train.replace(2, 0)
    y_train = y_train.replace(3, 0)

    y_test = y_test.replace(1, 0)
    y_test = y_test.replace(2, 0)
    y_test = y_test.replace(3, 0)

    # then replace 4 with "1"
    y_train = y_train.replace(4, 1)
    y_test = y_test.replace(4, 1)

    # print('y_train', y_train)
    print('y_train_number_after_change_label', y_train.value_counts())

# not address imbalance
#     X_train_imbalanced, y_train_imbalanced = X_train, np.array(y_train)

# address imbalance
    cc = SMOTETomek(random_state=42) # ,sampling_strategy={1:num_control}
#     cc = RandomOverSampler(random_state=42)
#     cc = ClusterCentroids(random_state=42)

    X_train_imbalanced, y_train_imbalanced = cc.fit_sample(X_train, y_train)


# load the classifier
#     with open(file_path + classifier_flag + '_cv_fold_' + str(n_fold) + '_classifier.pkl', 'rb') as fid:
#         classifier = pkl.load(fid)

# GridSearchCV
    classifier = RandomForestClassifier(random_state=seed)
    params = {'n_estimators': list(range(10, 120, 10)),  # list(range(20,120,10)),
              'max_depth': list(range(2, 10, 2)),
              # 'criterion': ['gini', 'entropy'],
              #  'min_samples_split': list(range(2,10,2)),
              # 'class_weight': ['balanced'],  # , None
              # 'max_leaf_nodes': list(range(2,10,2)),
              # 'max_features': list(range(2,20,2)),
              # 'min_samples_leaf': list(range(2,20,2))
              }
    grid = GridSearchCV(classifier, param_grid=params, scoring='roc_auc', cv=5, n_jobs=4)
    grid.fit(X_train_imbalanced, y_train_imbalanced)
    scores = pd.DataFrame(grid.cv_results_)
    # print('scores',scores)
    grid_accs = scores['mean_test_score'].values.tolist()
    best_auc = grid.best_score_
    best_auc_std = scores.loc[grid.best_index_, 'std_test_score']
    # print(best_auc_std)

    classifier = grid.best_estimator_
#
    print('Best by searching: %s, Std: %s' % (best_auc, best_auc_std))
    best_train_auc= best_auc
    print(grid.best_params_)

# training classifier
#     classifier = RandomForestClassifier(random_state =seed,n_estimators = 100,max_depth =8,class_weight ='balanced')
    classifier.fit(X_train_imbalanced, y_train_imbalanced)

# save the classifier
    with open(file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_classifier.pkl', 'wb') as fid:
        pkl.dump(classifier, fid)

    y_proba = classifier.predict_proba(X_test)[:, 1]  #
    # y_predic = classifier.predict(X_test)
    # print('y_predic',y_predic)

    test_result = y_proba
    aucroc = roc_auc_score(y_test, y_proba)
    print('aucroc', aucroc)

    # precision-recall auc
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    aucpr = auc(recall, precision)
    print('aucpr', aucpr)

#  plot feature importance
    X_data = X_train_imbalanced
    feature_cols = feature_cols

    fig, ax = plt.subplots(figsize=(15, 12))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.08, hspace=.5, wspace=.15)

    importances = classifier.feature_importances_
    # print('importances',importances)
    num_features = len(importances)
    indices = np.argsort(abs(importances))[::-1]
    plt.title("Feature importance")
    plt.barh(list(range(X_data.shape[1]))[:num_features], list(importances[indices])[:num_features], align="center")
    plt.yticks(list(range(X_data.shape[1]))[:num_features], (np.array(feature_cols)[indices]).tolist()[:num_features])

    # plt.show()
    fig_name = file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.png'
    fig.savefig(fig_name)
    plt.close('all')

    # print(importances.shape)
    importances_df = pd.DataFrame(importances.reshape((1,-1)),columns=feature_cols)
    importances_df.to_csv(file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.csv',index=False,header=True)

    return test_result

def build_classifier_1(X_train,y_train,X_test,y_test,feature_cols,classifier_flag, n_fold,file_path): #X[train], y[train],X[test]
# change label    so as to the objective case is "1", the objective of change label is to keep  classifier.predict_proba(X_text)[:, 1]   "1" is case
    print('---------------------this is classifier_1----------------------')
    print('y_train_number_before_change_label',y_train.value_counts())
    # replace 2,3 with 0,
    y_train = y_train.replace(2, 0)
    y_train = y_train.replace(3, 0)
    y_test = y_test.replace(2, 0)
    y_test = y_test.replace(3, 0)

    # print('y_train', y_train)
    print('y_train_number_after_change_label', y_train.value_counts())

    # load the classifier
    # with open(file_path + classifier_flag + '_cv_fold_' + str(n_fold) + '_classifier.pkl', 'rb') as fid:
    #     classifier = pkl.load(fid)


    # address imbalance
    # cc = SMOTETomek(random_state=42)  # ,sampling_strategy={1:num_control}
    # cc = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
#     cc = RandomOverSampler(random_state=42)
#     cc = ClusterCentroids(random_state=42)
#
#     X_train_imbalanced, y_train_imbalanced = cc.fit_sample(X_train, y_train)
#     X_train, y_train = X_train_imbalanced, y_train_imbalanced


# GridSearchCV
    classifier = RandomForestClassifier(random_state=seed)
    params = {'n_estimators': list(range(10, 120, 10)),  # list(range(20,120,10)),
              'max_depth': list(range(2, 10, 2)),
              # 'criterion': ['gini', 'entropy'],
              #  'min_samples_split': list(range(2,10,2)),
              'class_weight': ['balanced'],  # , None
              # 'max_leaf_nodes': list(range(2,10,2)),
              # 'max_features': list(range(2,20,2)),
              # 'min_samples_leaf': list(range(2,20,2))
              }
    grid = GridSearchCV(classifier, param_grid=params, scoring='roc_auc', cv=5, n_jobs=4)
    grid.fit(X_train, y_train)
    scores = pd.DataFrame(grid.cv_results_)
    # print('scores',scores)
    grid_accs = scores['mean_test_score'].values.tolist()
    best_acc = grid.best_score_
    best_acc_std = scores.loc[grid.best_index_, 'std_test_score']
    # print(best_auc_std)
    classifier = grid.best_estimator_
    print('Best by searching: %s, Std: %s' % (best_acc, best_acc_std))
    print(grid.best_params_)

# training classifier
#     classifier = RandomForestClassifier(random_state =seed,n_estimators = 100,max_depth =8,class_weight ='balanced')
    classifier.fit(X_train, y_train)
    y_proba = classifier.predict_proba(X_test)[:, 1]  #
    test_result = y_proba

    # print('y_proba',y_proba)
    aucroc = roc_auc_score(y_test, y_proba)
    print('aucroc', aucroc)

    # precision-recall auc
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    # print('precision',precision)
    # print('recall',recall)
    aucpr = auc(recall, precision)
    print('aucpr', aucpr)

# save the classifier
    with open(file_path + classifier_flag + '_cv_fold_' + str(n_fold) + '_classifier.pkl', 'wb') as fid:
        pkl.dump(classifier, fid)

#  plot feature importance
    X_data = X_train
    feature_cols = feature_cols

    fig, ax = plt.subplots(figsize=(15, 12))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.08, hspace=.5, wspace=.15)

    importances = classifier.feature_importances_

    # print('importances',importances)
    num_features = len(importances)
    indices = np.argsort(abs(importances))[::-1] #
    plt.title("Feature importance")
    plt.barh(list(range(X_data.shape[1]))[:num_features], list(importances[indices])[:num_features], align="center")
    plt.yticks(list(range(X_data.shape[1]))[:num_features], (np.array(feature_cols)[indices]).tolist()[:num_features])

    # plt.show()
    fig_name = file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.png'
    fig.savefig(fig_name)
    plt.close('all')

    # print(importances.shape)
    importances_df = pd.DataFrame(importances.reshape((1,-1)),columns=feature_cols)
    importances_df.to_csv(file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.csv',index=False,header=True)

    return test_result

def build_classifier_2(X_train,y_train,X_test,y_test,feature_cols,classifier_flag, n_fold,file_path): #X[train], y[train],X[test]
# change label    so as to the objective case is "1", the objective of change label is to keep  classifier.predict_proba(X_text)[:, 1]   "1" is case
    print('---------------------this is classifier_2----------------------')
    print('y_train_number_before_change_label',y_train.value_counts())
    # first replace 1,3 with 0, then replace 2 with 1
    y_train = y_train.replace(1, 0)
    y_train = y_train.replace(3, 0)
    y_test = y_test.replace(1, 0)
    y_test = y_test.replace(3, 0)

    # then replace 2 with 1
    y_train = y_train.replace(2, 1)
    y_test = y_test.replace(2, 1)

    print('y_train_number_after_change_label', y_train.value_counts())
    # load the classifier
    # with open(file_path + classifier_flag + '_cv_fold_' + str(n_fold) + '_classifier.pkl', 'rb') as fid:
    #     classifier = pkl.load(fid)


    # address imbalance
    cc = SMOTETomek(random_state=42) # ,sampling_strategy={1:num_control}
    # cc = RandomOverSampler(random_state=42)
    # cc = ClusterCentroids(random_state=42)

    X_train_imbalanced, y_train_imbalanced = cc.fit_sample(X_train, y_train)
    X_train, y_train = X_train_imbalanced, y_train_imbalanced

# GridSearchCV
    classifier = RandomForestClassifier(random_state=seed)
    params = {'n_estimators': list(range(10, 120, 10)),  # list(range(20,120,10)),
              'max_depth': list(range(2, 10, 2)),
              # 'criterion': ['gini', 'entropy'],
              #  'min_samples_split': list(range(2,10,2)),
              'class_weight': ['balanced'],  # , None
              # 'max_leaf_nodes': list(range(2,10,2)),
              # 'max_features': list(range(2,20,2)),
              # 'min_samples_leaf': list(range(2,20,2))
              }
    grid = GridSearchCV(classifier, param_grid=params, scoring='roc_auc', cv=5, n_jobs=4)
    grid.fit(X_train, y_train)
    scores = pd.DataFrame(grid.cv_results_)
    # print('scores',scores)
    grid_accs = scores['mean_test_score'].values.tolist()
    best_acc = grid.best_score_
    best_acc_std = scores.loc[grid.best_index_, 'std_test_score']
    # print(best_auc_std)

    classifier = grid.best_estimator_

    print('Best by searching: %s, Std: %s' % (best_acc, best_acc_std))
    print(grid.best_params_)

# training classifier
#     classifier = RandomForestClassifier(random_state =seed,n_estimators = 100,max_depth =8,class_weight ='balanced')
    classifier.fit(X_train, y_train)

    y_proba = classifier.predict_proba(X_test)[:, 1]  #

    test_result = y_proba

    # print('y_proba',y_proba)
    aucroc = roc_auc_score(y_test, y_proba)
    print('aucroc', aucroc)

    # precision-recall auc
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    # print('precision',precision)
    # print('recall',recall)
    aucpr = auc(recall, precision)
    print('aucpr', aucpr)

    # # save the classifier
    with open(file_path + classifier_flag + '_cv_fold_' + str(n_fold) + '_classifier.pkl', 'wb') as fid:
        pkl.dump(classifier, fid)

#  plot feature importance
    X_data = X_train
    feature_cols = feature_cols

    fig, ax = plt.subplots(figsize=(15, 12))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.08, hspace=.5, wspace=.15)

    importances = classifier.feature_importances_

    # print('importances',importances)
    num_features = len(importances)
    indices = np.argsort(abs(importances))[::-1] #
    plt.title("Feature importance")
    plt.barh(list(range(X_data.shape[1]))[:num_features], list(importances[indices])[:num_features], align="center")
    plt.yticks(list(range(X_data.shape[1]))[:num_features], (np.array(feature_cols)[indices]).tolist()[:num_features])

    # plt.show()
    fig_name = file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.png'
    fig.savefig(fig_name)
    plt.close('all')

    # print(importances.shape)
    importances_df = pd.DataFrame(importances.reshape((1,-1)),columns=feature_cols)
    importances_df.to_csv(file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.csv',index=False,header=True)

    return test_result

def build_classifier_3(X_train,y_train,X_test,y_test,feature_cols,classifier_flag, n_fold,file_path): #X[train], y[train],X[test]
# change label    so as to the objective case is "1", the objective of change label is to keep  classifier.predict_proba(X_text)[:, 1]   "1" is case
    print('---------------------this is classifier_3----------------------')
    print('y_train_number_before_change_label',y_train.value_counts())
    # first replace 1,2 with 0, then replace 3, with 1
    y_train = y_train.replace(1, 0)
    y_train = y_train.replace(2, 0)
    y_test = y_test.replace(1, 0)
    y_test = y_test.replace(2, 0)

    # then replace 3 with 1,
    y_train = y_train.replace(3, 1)
    y_test = y_test.replace(3, 1)

    # print('y_train', y_train)
    print('y_train_number_after_change_label', y_train.value_counts())

# not address imbalance
#     X_train_imbalanced, y_train_imbalanced = X_train, np.array(y_train)


    # address imbalance
    # cc = SMOTETomek(random_state=42) # ,sampling_strategy={1:num_control}
    cc = RandomOverSampler(random_state=42)
    # cc = ClusterCentroids(random_state=42)

    X_train_imbalanced, y_train_imbalanced = cc.fit_sample(X_train, y_train)

# GridSearchCV
    classifier = RandomForestClassifier(random_state=seed)
    params = {'n_estimators': list(range(10, 120, 10)),  # list(range(20,120,10)),
              'max_depth': list(range(2, 10, 2)),
              # 'criterion': ['gini', 'entropy'],
              #  'min_samples_split': list(range(2,10,2)),
              'class_weight': ['balanced'],  # , None
              # 'max_leaf_nodes': list(range(2,10,2)),
              # 'max_features': list(range(2,20,2)),
              # 'min_samples_leaf': list(range(2,20,2))
              }
    grid = GridSearchCV(classifier, param_grid=params, scoring='roc_auc', cv=5, n_jobs=4)
    grid.fit(X_train_imbalanced, y_train_imbalanced)
    scores = pd.DataFrame(grid.cv_results_)
    # print('scores',scores)
    grid_accs = scores['mean_test_score'].values.tolist()
    best_auc = grid.best_score_
    best_auc_std = scores.loc[grid.best_index_, 'std_test_score']
    # print(best_auc_std)

    classifier = grid.best_estimator_

    print('Best by searching: %s, Std: %s' % (best_auc, best_auc_std))
    best_train_auc= best_auc
    print(grid.best_params_)

    # load the classifier
    # with open(file_path + classifier_flag + '_cv_fold_' + str(n_fold) + '_classifier.pkl', 'rb') as fid:
    #     classifier = pkl.load(fid)

# training classifier
#     classifier = RandomForestClassifier(random_state =seed,n_estimators = 100,max_depth =8,class_weight ='balanced')
    classifier.fit(X_train_imbalanced, y_train_imbalanced)

    y_proba = classifier.predict_proba(X_test)[:, 1] 
    test_result = y_proba

    # print('y_proba',y_proba)
    aucroc = roc_auc_score(y_test, y_proba)
    print('aucroc', aucroc)

    # precision-recall auc
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    # print('precision',precision)
    # print('recall',recall)
    # print('thresholds',thresholds)

    aucpr = auc(recall, precision)
    print('aucpr', aucpr)

    # save the classifier
    with open(file_path + classifier_flag + '_cv_fold_' + str(n_fold) + '_classifier.pkl', 'wb') as fid:
        pkl.dump(classifier, fid)

##  plot feature importance
    X_data = X_train_imbalanced
    feature_cols = feature_cols

    fig, ax = plt.subplots(figsize=(15, 12))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.08, hspace=.5, wspace=.15)

    importances = classifier.feature_importances_

    # print('importances',importances)
    num_features = len(importances)
    indices = np.argsort(abs(importances))[::-1] #
    plt.title("Feature importance")
    plt.barh(list(range(X_data.shape[1]))[:num_features], list(importances[indices])[:num_features], align="center")
    plt.yticks(list(range(X_data.shape[1]))[:num_features], (np.array(feature_cols)[indices]).tolist()[:num_features])

    # plt.show()
    fig_name = file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.png'
    fig.savefig(fig_name)
    plt.close('all')

    # print(importances.shape)
    importances_df = pd.DataFrame(importances.reshape((1,-1)),columns=feature_cols)
    importances_df.to_csv(file_path+classifier_flag+'_cv_fold_'+str(n_fold)+'_feature_importance_aucroc'+str(aucroc)+'_aucpr_'+str(aucpr)+'.csv',index=False,header=True)

    return test_result

def change_label(data_df, case_flag):
    data_df = data_df

    if case_flag == 0:
        # first change case label 0->4;
        data_df.loc[data_df['group'] == 0, 'group'] = 4

        # then change control lable 1->0, 2->0, 3->0
        data_df.loc[data_df['group'] == 1, 'group'] = 0
        data_df.loc[data_df['group'] == 2, 'group'] = 0
        data_df.loc[data_df['group'] == 3, 'group'] = 0

        # third change case label 4->1;  "1" is fouced.
        data_df.loc[data_df['group'] == 4, 'group'] = 1

    if case_flag == 1:
        # change label, 0->0, 2->0, 3->0; ------1 vs rest
        data_df.loc[data_df['group'] == 0, 'group'] = 0
        data_df.loc[data_df['group'] == 2, 'group'] = 0
        data_df.loc[data_df['group'] == 3, 'group'] = 0
    if case_flag == 2:
        # first change control lable 0->0, 1->0, 3->0
        data_df.loc[data_df['group'] == 0, 'group'] = 0
        data_df.loc[data_df['group'] == 1, 'group'] = 0
        data_df.loc[data_df['group'] == 3, 'group'] = 0

        # then change case label 2->1;  "1" is fouced.
        data_df.loc[data_df['group'] == 2, 'group'] = 1

    if case_flag == 3:
        # first change control lable 0->0, 1->0, 2->0
        data_df.loc[data_df['group'] == 0, 'group'] = 0
        data_df.loc[data_df['group'] == 1, 'group'] = 0
        data_df.loc[data_df['group'] == 2, 'group'] = 0

        # then change case label 3->1;  "1" is fouced.
        data_df.loc[data_df['group'] == 3, 'group'] = 1

    return data_df

def compute_sem_ci(d_type, d_mean, d_std):
    sem = d_std / math.sqrt(5)
    z = 1.960  # (confidence 0.95)
    CI_h = z * sem
    CI_low = d_mean - CI_h
    CI_high = d_mean + CI_h

    d_mean = float('%.4f' % d_mean)
    CI_low = float('%.4f' % CI_low)
    CI_high = float('%.4f' % CI_high)
    CI_h = float('%.4f' % CI_h)
    d_std = float('%.4f' % d_std)
    sem = float('%.4f' % sem)

    result = {
        'type': d_type,
        'mean': d_mean,
        'ci_lower': CI_low,
        'ci_upper': CI_high,
        'ci_h': CI_h,
        'std': d_std,
        'sem': sem}

    return result

def mean_confidence_interval(data, confidence=0.95):
    # https://www.mathsisfun.com/data/confidence-interval-calculator.html
    if confidence==0.95:
        z = 1.960

    a = 1.0 * np.array(data)
    n = len(a)
    a_mean = np.mean(a)
    a_std = np.std(a)

    CI_h = z * a_std / math.sqrt(n)
    CI_low = a_mean - CI_h
    CI_high = a_mean + CI_h

    return CI_low, CI_high,CI_h

def normalize(all_conti_x_fitted, all_categ_x):

    conti_columns = all_conti_x_fitted.columns.values
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    all_conti_x_fitted = scaler.fit_transform(all_conti_x_fitted)
    all_conti_x_fitted = pd.DataFrame(all_conti_x_fitted,columns=conti_columns)
    # all_conti_x_fitted.to_csv(file_path + 'xx_conti_0.csv')

    onehot_enc = OneHotEncoder(sparse=False)  # dense format
    all_categ_x_fitted = onehot_enc.fit_transform(all_categ_x)
    columns = list(onehot_enc.categories_)
    column_list = []
    for term in columns:
        term = term.tolist()
        column_list+=term
    # print(column_list)
    all_categ_x_fitted=pd.DataFrame(all_categ_x_fitted,columns=column_list)

    # all_categ_x_fitted.to_csv(file_path+'xx_0.csv')

    return all_conti_x_fitted, all_categ_x_fitted

def normalize_rf(all_conti_x_fitted, all_categ_x):

    all_conti_x_fitted = all_conti_x_fitted

    onehot_enc = OneHotEncoder(sparse=False)  # dense format
    all_categ_x_fitted = onehot_enc.fit_transform(all_categ_x)
    columns = list(onehot_enc.categories_)
    column_list = []
    for term in columns:
        term = term.tolist()
        column_list+=term
    # print(column_list)
    all_categ_x_fitted=pd.DataFrame(all_categ_x_fitted,columns=column_list)

    all_categ_x_fitted.to_csv(file_path+'xx_cat_rf_0.csv')

    return all_conti_x_fitted, all_categ_x_fitted


file_path_data = '/Users/xuzhenxing/Documents/Sepsis/prediction_data/'
file_path = '/Users/xuzhenxing/Documents/Sepsis/prediction_result/'
file_path_subscore = '/Users/xuzhenxing/Documents/Sepsis/'

# subscore_joint
# read subscore csv file
sofa_score = pd.read_csv(file_path_subscore + 'sofa.csv',index_col=False)
respiration_score = pd.read_csv(file_path_subscore + 'respiration_score.csv',index_col=False)
coagulation_score = pd.read_csv(file_path_subscore + 'coagulation_score.csv',index_col=False)
liver_score = pd.read_csv(file_path_subscore + 'liver_score.csv',index_col=False)
cardiovascular_score = pd.read_csv(file_path_subscore + 'cardiovascular_score.csv',index_col=False)
cns_score = pd.read_csv(file_path_subscore + 'cns_score.csv',index_col=False)
renal_score = pd.read_csv(file_path_subscore + 'renal_score.csv',index_col=False)

# subscore joint
sofa_score = pd.merge(sofa_score, respiration_score, how='left', on='icustay_id')
sofa_score = pd.merge(sofa_score, coagulation_score, how='left', on='icustay_id')
sofa_score = pd.merge(sofa_score, liver_score, how='left', on='icustay_id')
sofa_score = pd.merge(sofa_score, cardiovascular_score, how='left', on='icustay_id')
sofa_score = pd.merge(sofa_score, cns_score, how='left', on='icustay_id')
sofa_score = pd.merge(sofa_score, renal_score, how='left', on='icustay_id')

sofa_score = sofa_score.rename(columns={"icustay_id":"ICU_stay_ID"})

# read comorbidity, lab test, vital signs
lab_vital_df = pd.read_csv(file_path_data + 'NW_prediction_mat.csv',index_col=False)

# merge data sofa_score and lab_vital_df
data_df = pd.merge(lab_vital_df, sofa_score, how='left', on='ICU_stay_ID')
# get sofa 6 subscore (in first 6 hours) and rename, and extract lab_vital variables
data_df = data_df.rename(columns={"sofa_6": "SOFA_score",
                                  "respiration_score_6": "Respiration_score",
                                  "coagulation_score_6": "Coagulation_score",
                                  "liver_score_6": "Liver_score",
                                  "cardiovascular_score_6": "Cardiovascular_score",
                                  "cns_score_6": "CNS_score",
                                  "renal_score_6": "Renal_score"
                                  })

obj_feature_name = ['SOFA_score', 'Respiration_score', 'Coagulation_score', 'Liver_score', 'Cardiovascular_score','CNS_score', 'Renal_score',
             'Bands', 'CRP', 'Temperature', 'WBC', 'SO2', 'Pao2', 'Respiratory_rate',
             'Bicarbonate', 'Heart_rate', 'Lactate', 'Systolic_ABP', 'Troponin I','BUN', 'Creatinine',
            'ALT', 'AST', 'Bilirubin','GCS','Hemoglobin', 'INR', 'Platelet','Albumin',
            'Chloride', 'Glucose', 'Sodium', 'RDW', 'Lymphocyte_count','Lymphocyte_percent', 'BMI',
            'Age', 'Comorbidity_score']

data_df = data_df[['ICU_stay_ID','group']+obj_feature_name]
data_df = data_df.fillna(data_df.median())

# data_df.to_csv(file_path+'data_df.csv',index=False)
# print('len(data_df)',len(data_df))

data_df.to_csv(file_path+'data_df_ori.csv',index=False)

# remove outliers
# Note that, if there are outliers, the outlier is found based on their abnormal value, e.g., age ==300
outlier = []
# feature_cols = list(data_df.drop(['ICU_stay_ID', 'group'], axis=1).columns)
data_df = data_df.loc[~data_df['ICU_stay_ID'].isin(outlier)]
data_df = data_df.reset_index(drop=True)


# feature transformation
lab_vital_trans = ['ALT', 'AST', 'Bands', 'Bilirubin', 'BUN', 'Creatinine', 'CRP', 'Glucose', 'INR', 'Lactate',
                   'SO2', 'Systolic_ABP', 'Troponin I', 'WBC']

for fea in lab_vital_trans:
    data_df[fea] = np.log(data_df[fea] + 0.1)

# data_df.to_csv(file_path + 'data_df_transformation.csv',index=False)
data_df.to_csv(file_path+'data_df.csv',index=False)

X = data_df[obj_feature_name].values
y = data_df['group']
icu_id = data_df['ICU_stay_ID']

print('len X:', len(X))
print('len y:', len(y))
print('num_y_0', np.count_nonzero(y == 0))
print('num_y_1', np.count_nonzero(y == 1))
print('num_y_2', np.count_nonzero(y == 2))
print('num_y_3', np.count_nonzero(y == 3))

cv = KFold(n_splits=5, shuffle=True,random_state=seed)
n_fold = 0
acc_total = []

best_train_auc_total = []
aucroc_total = []
aucpr_total = []
pre_label_0_total = []
pre_label_1_total = []
pre_label_2_total = []
pre_label_3_total = []

rec_label_0_total = []
rec_label_1_total = []
rec_label_2_total = []
rec_label_3_total = []

feature_cols = obj_feature_name
all_metrics_result_df = pd.DataFrame(columns=['type','mean', 'ci_lower', 'ci_upper', 'ci_h', 'std', 'sem'])

for train, test in cv.split(X):
    print('this is the result of fold:',n_fold)

    X_ori = X # X[train], X  # using X, y as input to obtain the feature importance
    y_ori = y # y[train], y  # using X, y as input to obtain the feature importance

    C_0_result = build_classifier_0(X_ori, y_ori, X[test], y[test], feature_cols, 'classifier_0', n_fold,file_path)
    C_1_result = build_classifier_1(X_ori, y_ori, X[test], y[test], feature_cols, 'classifier_1', n_fold,file_path)
    C_2_result = build_classifier_2(X_ori, y_ori, X[test], y[test], feature_cols, 'classifier_2', n_fold,file_path)
    C_3_result = build_classifier_3(X_ori, y_ori, X[test], y[test], feature_cols, 'classifier_3', n_fold,file_path)

    C_result_proba = np.array([C_0_result,C_1_result,C_2_result,C_3_result])
    # # C_result_proba = np.array([C_1_result, C_2_result])
    C_result_proba /= (np.sum(C_result_proba, axis=0))
    
    # # C_result_proba_max it the probality
    # # C_result_proba_max_index is the label with max probality
    C_result_proba_max = np.max(C_result_proba, axis=0)
    C_result_proba_max_index = np.argmax(C_result_proba, axis=0)
    
    y_predic = C_result_proba_max_index

    y_test_tem = list(y[test])
    y_pre_tem = list(y_predic)

    print('y_test_tem',y_test_tem)
    print('y_pre_tem',y_pre_tem)

    predict_result = classification_report(y_test_tem, y_pre_tem,output_dict=True)

    print(classification_report(y_test_tem, y_pre_tem))

    acc_score = predict_result['accuracy']
    acc_total.append(acc_score)
    print('acc_score',acc_score)

    pre_score_0, rec_score_0 = predict_result['0']['precision'], predict_result['0']['recall']
    pre_score_1, rec_score_1 = predict_result['1']['precision'], predict_result['1']['recall']
    pre_score_2, rec_score_2 = predict_result['2']['precision'], predict_result['2']['recall']
    pre_score_3, rec_score_3 = predict_result['3']['precision'], predict_result['3']['recall']

    pre_label_0_total.append(pre_score_0)
    pre_label_1_total.append(pre_score_1)
    pre_label_2_total.append(pre_score_2)
    pre_label_3_total.append(pre_score_3)

    rec_label_0_total.append(rec_score_0)
    rec_label_1_total.append(rec_score_1)
    rec_label_2_total.append(rec_score_2)
    rec_label_3_total.append(rec_score_3)

    n_fold = n_fold + 1

# print acc
acc_total_mean = np.mean(np.array(acc_total))
acc_total_std = np.std(np.array(acc_total))
print('acc_total_mean',acc_total_mean)
print('acc_total_std',acc_total_std)

mean_pre_0 = np.mean(np.array(pre_label_0_total))
mean_pre_1 = np.mean(np.array(pre_label_1_total))
mean_pre_2 = np.mean(np.array(pre_label_2_total))
mean_pre_3 = np.mean(np.array(pre_label_3_total))

mean_rec_0 = np.mean(np.array(rec_label_0_total))
mean_rec_1 = np.mean(np.array(rec_label_1_total))
mean_rec_2 = np.mean(np.array(rec_label_2_total))
mean_rec_3 = np.mean(np.array(rec_label_3_total))

std_pre_0 = np.std(np.array(pre_label_0_total))
std_pre_1 = np.std(np.array(pre_label_1_total))
std_pre_2 = np.std(np.array(pre_label_2_total))
std_pre_3 = np.std(np.array(pre_label_3_total))

std_rec_0 = np.std(np.array(rec_label_0_total))
std_rec_1 = np.std(np.array(rec_label_1_total))
std_rec_2 = np.std(np.array(rec_label_2_total))
std_rec_3 = np.std(np.array(rec_label_3_total))

print('this is the result of label 0:')
print('mean_pre_0,mean_rec_0', mean_pre_0, mean_rec_0)
print('std_pre_0,std_rec_0', std_pre_0, std_rec_0)

print('this is the result of label 1:')
print('mean_pre_1,mean_rec_1', mean_pre_1, mean_rec_1)
print('std_pre_1,std_rec_1', std_pre_1, std_rec_1)

print('this is the result of label 2:')
print('mean_pre_2,mean_rec_2', mean_pre_2, mean_rec_2)
print('std_pre_2,std_rec_2', std_pre_2, std_rec_2)

print('this is the result of label 3:')
print('mean_pre_3,mean_rec_3', mean_pre_3, mean_rec_3)
print('std_pre_3,std_rec_3', std_pre_3, std_rec_3)

# compute sem, CI_h, CI_low,...
type_metrics = ['class_0_precision','class_0_recall','class_1_precision','class_1_recall',
                'class_2_precision','class_2_recall','class_3_precision','class_3_recall','Total_class_accuracy']
all_metrics_mean = [mean_pre_0, mean_rec_0, mean_pre_1, mean_rec_1, mean_pre_2, mean_rec_2, mean_pre_3, mean_rec_3, acc_total_mean]
all_metrics_std = [std_pre_0, std_rec_0, std_pre_1, std_rec_1, std_pre_2, std_rec_2, std_pre_3, std_rec_3, acc_total_std]
for i in range(len(type_metrics)):
    result_metrics = compute_sem_ci(type_metrics[i],all_metrics_mean[i],all_metrics_std[i])
    all_metrics_result_df = all_metrics_result_df.append(result_metrics, ignore_index=True, sort=False)
    
# save all metrics results
all_metrics_result_df.to_csv(file_path+'all_metrics_prediction_result.csv',index=False)






