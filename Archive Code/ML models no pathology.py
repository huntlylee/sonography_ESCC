# -*- coding: utf-8 -*-
"""
for model 1 development and testing

@author: liy45
"""

import numpy as np
import matplotlib.pyplot as plt

# import os, os.path
import pandas as pd
# import math


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils import class_weight

# from sklearn.naive_bayes import ComplementNB, MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# from keras.callbacks import EarlyStopping

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score, roc_auc_score, make_scorer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
# import shap

from sklearn.inspection import permutation_importance

from collections import Counter
from statsmodels.stats.contingency_tables import cochrans_q 


''' 1. Data preprocessing'''

# Read the file 
file_path = r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Neck data.xlsx'

df_original = pd.read_excel(file_path, index_col=None)

unwanted = ['ID', '现病史1','ESCC_family','身高', '体重', 'BMI', '肿瘤位置(1=upper,2=middle,3=lower)',
                  'Grade', '新辅助', '101R(4A)', '101L(3A)', '102104R(6)', '102104L(5)', 'LN_harvest', 'LN+',
                  '病理', '超声', '超声颈部LNM']

df_original.drop(unwanted, axis=1, inplace=True)

df_original.rename(columns={"超声LN最大径mm": "Diameter", 
                            "探及低回声": "Hypoecho",
                            "形态（0=规则，1=饱满/不规则/淋巴结融合）": "Shape",
                            "淋巴门结构（0=完整，1=模糊）": "Hilum",
                            '回声是否均匀有无血流（0=均匀无血流，1=不均匀有血流）': 'Bloodflow',
                            '颈部LN' : 'Neck_LN', '胸部除去左右喉返(0=neg,1=pos,3=unk)': 'Chest',
                            '腹部(0=neg,1=pos,3=unk)': 'Abdomen', '胸内右喉返神经(4B)': 'R-RLN',
                            '胸内左喉返神经(3B)': 'L-RLN',
                            '肿瘤Size': 'Size'
                            }, inplace=True)

def rln(row):
    if row['R-RLN'] == 1 or row['L-RLN'] == 1:
        return 1
    else:
        return 0

df_original.drop(df_original[(df_original['R-RLN'] == 3) | (df_original['L-RLN'] ==3)].index, inplace=True)
df_original['RLN'] = df_original.apply(rln, axis=1)


def ultrasound_abnormal(row):
    if row['Hypoecho'] == 1 or row['Shape'] == 1 or row['Hilum'] == 1 or row['Bloodflow'] == 1:
        return 1
    else:
        return 0
    
df_original['Ultrasound'] = df_original.apply(ultrasound_abnormal, axis=1)

df_original.drop(['R-RLN', 'L-RLN', 'Ultrasound', 'Shape', 'Hilum', 'Bloodflow', 'Hypoecho'], axis=1, inplace=True)

df_original.dropna(axis=0, how='any', inplace=True)


def descriptive_table(df, show_missing_only = True):
        # zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isna().sum()
        mis_val_percent = 100 * df.isna().sum() / len(df)
        min_val = df.min()
        max_val = df.max()
        mz_table = pd.concat([mis_val, mis_val_percent, min_val, max_val], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Missing Values', 1 : 'Missing Values %', 2 : 'Min Value', 3: 'Max Value'})
        # mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        # mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        
        missing_count = len(mz_table[mz_table.iloc[:,1] != 0])
        
        if show_missing_only == True:
            mz_table = mz_table[mz_table.iloc[:,1] != 0].sort_values('Missing Values %', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
               "There are " + str(missing_count) + " columns that have missing values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table

stat_original = descriptive_table(df_original, show_missing_only = False)


''' gen training/validation dataset'''

preop = ['Gender', 'Age', 'Smokers', 'Alcohol', 'Cancer_family', 'Distance', 'T', 'Size', 'Diameter']

pathology = ['Chest', 'Abdomen', 'RLN']

target = ['Neck_LN']

features = [e for e in list(df_original) if e in preop]

x = df_original.loc[:, features]
y = df_original.loc[:, target]


stat_original.to_excel(r'C:\Users\liy45\Desktop\stats.xlsx')



'''check correlation'''

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.show()
    
correlation_heatmap(x)

# correlation_heatmap(df_original)
def create_descriptive_table(x):
    
    first_item_list, second_item_list = [], []
    feature_num_list, feature_std_list = [], []
    
    feature_list = list(x.columns)
    
    for feature in feature_list:
        
        a = Counter(x[feature])
        
        if len(a) >= 10: # conintuous:
            first_item_list.append(feature)
            second_item_list.append(feature)
            feature_num_list.append(x[feature].mean())
            feature_std_list.append(x[feature].std())            
        else:
            total_n = sum(a.values())
            for e in list(a.keys()):
                first_item_list.append(feature)
                second_item_list.append(e)
                feature_num_list.append(a[e])
                feature_std_list.append(a[e]/total_n)
    
    df = pd.DataFrame(list(zip(first_item_list, second_item_list, feature_num_list, feature_std_list)),
                      columns=['Features', 'Class', 'Count or mean', '% or SD']
                      )
    
    return df

stat_x = create_descriptive_table(x)

stat_x.to_excel(r'C:\Users\liy45\Desktop\stats_all.xlsx', index = False)



''' 2. Hyperparameter tuning'''

def param_tune_single(ml_model = 'RF', test_param = 'n_estimators', param_range = None):
       
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 3214, stratify=y)
    
    train_results = []
    test_results = []
    param_grid = {test_param: param_range}
    
    for param in param_grid[test_param]:
        
        dict_temp = {test_param: param}
        if ml_model == 'RF':
            clf = RandomForestClassifier(random_state = 10, n_jobs=-1, class_weight = 'balanced', criterion='gini', **dict_temp)

        elif ml_model == 'SVM':
            clf = SVC(probability=True, class_weight='balanced', max_iter = 100000, kernel = 'rbf', gamma = 'scale', **dict_temp)
     
        elif ml_model == 'KNN':
            clf = KNeighborsClassifier(weights = 'distance', metric = 'minkowski',  **dict_temp)
              
        elif ml_model == 'XGB':
            clf = xgb.XGBClassifier(subsample=0.5, random_state = 20, n_jobs = -1, eta = 0.1, gamma = 0.1, objective = 'binary:logistic', **dict_temp) # around 70
    
        elif ml_model == 'LGB':
            clf = lgb.LGBMClassifier(subsample=0.5, objective  = 'binary', random_state = 20, n_jobs = -1, learning_rate = 0.1,
                                     silent= True, **dict_temp) # around 70
    
        else:
            clf = LogisticRegression(penalty='l2', class_weight = 'balanced', max_iter = 100000, **dict_temp, solver = 'liblinear') #,
        
        print('training {} at {}...'.format(test_param, param))
        clf.fit(x_train.values, np.ravel(y_train.values))
        train_pred = clf.predict(x_train.values)
        fpr, tpr, thresholds = roc_curve(np.ravel(y_train), train_pred)
        roc_auc = auc(fpr, tpr)
        train_results.append(roc_auc)
        y_pred = clf.predict(x_test.values)
        fpr, tpr, thresholds = roc_curve(np.ravel(y_test), y_pred)
        roc_auc = auc(fpr, tpr)
        test_results.append(roc_auc)
    
    line1, = plt.plot(param_grid[test_param], train_results, 'b', label='Train AUC')
    line2, = plt.plot(param_grid[test_param], test_results, 'r', label='Test AUC')
    plt.legend()
    plt.ylabel('AUC score')
    plt.xlabel(test_param)
    plt.show()
    
param_tune_single(ml_model = 'LR', test_param = 'C', param_range = list(map(lambda x:pow(2, x), list(range(-2, 5, 1)))))



''' define classifier and grid search target'''

def select_clf(ml_model = 'SVM'):
   
    if ml_model == 'SVM':
        clf = SVC(probability=True, class_weight='balanced', max_iter = 100000, kernel = 'rbf', gamma = 'scale')
        ''' probability: This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation, 
            and predict_proba may be inconsistent with predict'''
        param_grid = {#'kernel': ['linear', 'poly', 'rbf'], #'sigmoid'
                      'C': list(map(lambda x:pow(2, x), list(range(-3, 5, 1)))),
                      # 'gamma': ['scale', 'auto']
                      }
        
    elif ml_model == 'RF':
        clf = RandomForestClassifier(random_state = 10, n_jobs=-1, class_weight = 'balanced', criterion='gini')
        param_grid = {# 'bootstrap': [True],
                      'max_depth': list(range(2, 7, 1)),
                      'max_features': list(range(2,16,3)),
                      'min_samples_leaf': list(range(6, 20, 3)),
                      'min_samples_split': list(range(20, 60, 10)),
                      'n_estimators': [16, 32, 64],
                      # 'class_weight' : ['balanced', 'balanced_subsample'],
                      # 'criterion': ['gini', 'entropy']
                      }
        
    elif ml_model == 'KNN':
        clf = KNeighborsClassifier(weights = 'distance', metric = 'minkowski')
        param_grid = {#'weights': ['uniform', 'distance'],
                      'n_neighbors': list(range(20, 30, 3)),
                      'leaf_size' : list(range(10, 25, 5)),
                      # 'metric': ['euclidean', 'manhattan', 'minkowski']
                      }   
        
    elif ml_model == 'XGB':
        clf = xgb.XGBClassifier(subsample=0.5, random_state = 20, n_jobs = -1, gamma = 0.1, eta = 0.1, objective = 'binary:logistic') # around 70
        param_grid = {#'eta' : [0.1, 0.2, 0.3, 0.4, 0.5] ,
                      'max_depth' : list(range(5, 15, 3)),
                      'min_child_weight' : list(range(8, 13, 1)),
                      # 'gamma' : [ 0.0, 0.1, 0.2, 0.3],
                      'colsample_bytree' : list(np.linspace(0.35, 0.45, 3, endpoint=True)),
                      'n_estimators':  [16, 32, 64]
                      }
        
    elif ml_model == 'LGB':
        clf = lgb.LGBMClassifier(subsample=0.5, objective  = 'binary', random_state = 20, n_jobs = -1, learning_rate = 0.1,
                                 silent = True) 
        param_grid = {#'learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5],
                      'num_leaves' : list(range(20, 30, 3)),
                      'min_child_samples': list(range(80, 100, 5)),
                      #'gamma' : [0.0, 0.1, 0.2, 0.3],
                      'colsample_bytree' : list(np.linspace(0.1, 0.3, 3, endpoint=True)),
                      'n_estimators':  [32, 64, 128]
                      }
    else:
        clf = LogisticRegression(class_weight = 'balanced', max_iter = 1000000, solver = 'liblinear')
        param_grid = {
                    # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'C': list(map(lambda x:pow(2, x), list(range(-2, 4, 1))))
                      }

    return clf, param_grid


# def custom_auc(ground_truth, predictions):
#      # I need only one column of predictions["0" and "1"]. You can get an error here
#      # while trying to return both columns at once
#      fpr, tpr, _ = roc_curve(ground_truth, predictions)    
#      return auc(fpr, tpr)

# to be standart sklearn's scorer        
# my_auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)

# scorers = {
#     'precision_score': make_scorer(precision_score),
#     'recall_score': make_scorer(recall_score),
#     'accuracy_score': make_scorer(accuracy_score),
#     'auc_score': my_auc
# }

''' creat my own scoring function: npv for hyperparamenter tuning'''

def grid_search_wrapper(x_dev, y_dev, clf, scoring = 'PR'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    if scoring == 'PR':
        
        score_func = 'average_precision'
    
    elif scoring == 'ROC':
        
        score_func = 'roc_auc'
             
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
    
    grid_search = GridSearchCV(clf, param_grid, scoring = score_func, refit = False,
                           cv=skf, return_train_score=False, n_jobs=-1, verbose = 2)
    
    grid_search.fit(x_dev, np.ravel(y_dev))

    print('Best params for {}'.format(scoring))
    print(grid_search.best_params_)


    return grid_search


def train_test_model(x_train, y_train, x_val, ml_model):

    ml_model.fit(x_train, np.ravel(y_train))
    
    pred_prob = ml_model.predict_proba(x_val)
    
    return pred_prob


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]



''' execution of step 2: to get the best hyperparameters via GridSearchCV on the whole dataset'''

# x_dev, x_test, y_dev, y_test = scale_dataset(x, y, scale = 'None')

clf, param_grid = select_clf('LGB')
grid_search_result = grid_search_wrapper(x, y, clf, scoring = 'ROC') # tune hyperparameters to maximize scoring
print(grid_search_result.best_params_)
# grid_search_clf = grid_search_result.best_estimator_  




''' Best hyperparameters all features:
    Random Forest: {'max_depth': 6, 'max_features': 2, 'min_samples_leaf': 6, 'min_samples_split': 40, 'n_estimators': 32}
    KNN : {'leaf_size': 10, 'n_neighbors': 29}
    SVM : {'C': 16}
    Logistic regression: {'C': 0.25}
    XGBoost: {'colsample_bytree': 0.45, 'max_depth': 5, 'min_child_weight': 12, 'n_estimators': 64}
    LightGBM: {'colsample_bytree': 0.3, 'min_child_samples': 95, 'n_estimators': 32, 'num_leaves': 20}

    Best hyperparameters preop:
    Random Forest: {'max_depth': 2, 'max_features': 8, 'min_samples_leaf': 18, 'min_samples_split': 40, 'n_estimators': 64}
    KNN : {'leaf_size': 10, 'n_neighbors': 29}
    SVM : {'C': 2}
    Logistic regression: {'C': 0.5}
    XGBoost:  {'colsample_bytree': 0.45, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 32}
    LightGBM: {'colsample_bytree': 0.2, 'min_child_samples': 95, 'n_estimators': 64, 'num_leaves': 20}

'''



''' Step 3. cross-validation with the best hyperparameters 
    get the 95% threshold'''

def train_clf(x_dev, y_dev, ml_model = 'SVM'):
    
    
    if ml_model == 'SVM':
        clf = SVC(probability=True, class_weight='balanced', max_iter = 100000, kernel = 'rbf', gamma = 'scale', C = 2)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'Standard')
        
    elif ml_model == 'RF':
        clf = RandomForestClassifier(random_state = 10, n_jobs=-1, class_weight = 'balanced', criterion='gini', 
                                     max_depth = 2, max_features = 8, min_samples_leaf = 18, min_samples_split = 40,
                                     n_estimators = 64)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')

    elif ml_model == 'KNN':
        clf = KNeighborsClassifier(weights = 'distance', metric = 'minkowski', leaf_size = 10, n_neighbors = 29)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')
        
    
    else:
        clf = LogisticRegression(class_weight = 'balanced', max_iter = 1000000, solver = 'liblinear', C = 0.5)
        # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')

    x_train,x_val,y_train,y_val = train_test_split(x_dev, y_dev,test_size = 0.2, random_state = 10, shuffle = True, stratify = y_dev)    

    clf.fit(x_train, np.ravel(y_train))
 
    pred_score = clf.predict_proba(x_val)[:,1]
              
    fpr, tpr, auc_thresholds = roc_curve(y_val, pred_score)
    
    balanced_idx = np.argmax(tpr - fpr)
    balanced_threshold = auc_thresholds[balanced_idx] 
    
    # spe = 1-fpr
    # prevalence = np.count_nonzero(y_val)/len(y_val) # should be around 920/(5673+920)
    # npv = spe*(1-prevalence)/((1-tpr)*prevalence + spe*(1-prevalence))

    # max_npv_idx = np.argmax(npv>=0.90)
    # max_npv_threshold = auc_thresholds[max_npv_idx] 
    
    # y_test_score = clf.predict_proba(x_test)[:,1]
    
    # explainer = shap.TreeExplainer(clf)
    # shap_values = explainer.shap_values(x_test)

    # shap.summary_plot(shap_values, x_test, plot_type="bar")  
    

    return clf, balanced_threshold



def get_metrics_testset(y_true, y_scores, b_threshold):
    
    # y_scores =  y_pred[:,1]
    
    fpr, tpr, auc_thresholds = roc_curve(y_true, y_scores)
    
    roc_score = auc(fpr, tpr)
    
    ap = average_precision_score(y_true, y_scores)
    
    # balanced_idx = np.argmax(tpr - fpr)
    # balanced_threshold = auc_thresholds[balanced_idx] 
    
    # spe = 1-fpr
    # prevalence = np.count_nonzero(y_true)/len(y_true) # should be around 920/(5673+920)
    # npv = spe*(1-prevalence)/((1-tpr)*prevalence + spe*(1-prevalence))

    # max_npv_idx = np.argmax(npv>=0.95)
    # max_npv_threshold = auc_thresholds[max_npv_idx] 
             
    y_pred_lbl = adjusted_classes(y_scores, b_threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_lbl).ravel()
    
    accuracy = (tp+tn)/len(y_pred_lbl)

    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
                   
    pre = tp / (tp+fp)

    print('\nAt threshold {:.4f}\n'.format(b_threshold))
    print(pd.DataFrame(np.array([tp,fp,fn,tn]).reshape(2,2),
                       columns=['pos', 'neg'], 
                       index=['pred_pos', 'pred_neg']))
    
    print('\nAccuracy: {:.4f}\nSensitivity: {:.4f}\nSpecificity: {:.4f}\nPrecision: {:.4f}\nROC: {:.4f}\nAP: {:.4f}'.format(
            accuracy, tpr, tnr, pre, roc_score, ap))

    return tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap



''' Step 4. To get the permutation importance score'''

def plot_feature_importance(df, top = 5, title_text = None):
    
    # feature_importance = permutation_importance(clf, x_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=10)
    
    # df_fea_all = pd.DataFrame({'permutation importance score': feature_importance.importances_mean,
    #                        'error': 1.96 * feature_importance.importances_std}, index=list(x_test.columns))

    # if show_sig_only == True:
    if top is not None:   
        df_fea = df.sort_values('Mean').iloc[-top:]
    else:
        df_fea = df.sort_values('Mean')
        
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8)) 
 
    ax = df_fea.plot.barh(y='Mean', xerr = 'SEM',  color='#86bf91')
    
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    # Draw vertical axis lines
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
        
    ax.legend().set_visible(False)
    
    # Set x-axis label
    ax.set_xlabel("Feature importance score", labelpad=20, weight='bold', size=18)
    
    # Set y-axis label
    ax.set_ylabel("Feautures", labelpad=20, weight='bold', size=16)
    
    ax.set_title(title_text + ' - ' + ml_type, size = 20)
    
    


''' execution of step 3 and 4'''

ml_type = 'LR'

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12345)

thresh_list, acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list= [], [], [], [], [], [], []
tn_list, fp_list, fn_list, tp_list = [], [], [] ,[]
fold_list = []
df_dict = {}

feature_importance_list = []

fold_num = 1

for train_index, test_index in skf.split(x, y):
    
    x_dev, x_test = x.iloc[train_index], x.iloc[test_index]
    y_dev, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf, op_threshold = train_clf(x_dev.values, y_dev.values, ml_model = ml_type)
 
    y_test_score = clf.predict_proba(x_test.values)[:,1]   
    y_pred_lbl = [1 if y >= op_threshold else 0 for y in y_test_score]

    df_dict[fold_num] = pd.DataFrame(list(zip(test_index, y_test_score, y_pred_lbl, y_test.values.flatten())),
                         columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])
    
    tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap_score = get_metrics_testset(y_test.to_numpy(), y_test_score, op_threshold)
    
    fold_list.append(fold_num)
    thresh_list.append(op_threshold)
    acc_list.append(accuracy)
    tpr_list.append(tpr)
    tnr_list.append(tnr)
    pre_list.append(pre)
    auc_list.append(roc_score)
    ap_list.append(ap_score)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)
    
    feature_importance = permutation_importance(clf, x_test.values, y_test.values, scoring='roc_auc', n_repeats=50, random_state=10, n_jobs=-1) # scoring='neg_log_loss'

    result = pd.DataFrame(data = list(zip(fold_list, thresh_list, tn_list, fp_list, fn_list, tp_list, 
                                          acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list)),
                          columns= ['Fold', 'Threshold', 'TN', 'FP', 'FN', 'TP',
                                    'Accuracy', 'Recall', 'Specificity', 'Precision', 'ROC', 'AP'])

    feature_importance_list.append(feature_importance.importances_mean)
    
    fold_num +=1

df_result = pd.concat(df_dict).droplevel(1, axis=0).reset_index()  
df_result.rename(columns={'index': 'Fold'}, inplace=True) 

    
a = np.vstack(feature_importance_list)

df_importance = pd.DataFrame(list(zip(np.mean(a, axis = 0), np.std(a, axis = 0)/np.sqrt(len(a)))),
                             index=list(x_test.columns),
                             columns = ['Mean', 'SEM'])

df_result.to_excel(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), sheet_name='case', index = False)
with pd.ExcelWriter(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), engine='openpyxl', mode='a') as writer:  
    # df_result.to_excel(writer, sheet_name='case', index = False)
    result.to_excel(writer, sheet_name='sum', index=False)
    df_importance.to_excel(writer, sheet_name='importance', index=True) 
    
plot_feature_importance(df_importance, top = 20, title_text = target[0].upper())





''' new ml models: XGBoost, Light GBM'''

# dtrain = xgb.DMatrix(x_train, label=y_train)
# dval = xgb.DMatrix(x_val, label=y_val)
# dtest = xgb.DMatrix(x_test, label=y_test)

# param  = {'colsample_bytree': 0.3, 'eta': 0.1, 'gamma': 0.2, 'max_depth': 5, 'min_child_weight': 7}
# evallist = [(dval, 'eval'), (dtrain, 'train')]

''' Best hyperparameters all features:
    
    XGBoost: {'colsample_bytree': 0.45, 'max_depth': 5, 'min_child_weight': 12, 'n_estimators': 64}
    LightGBM: {'colsample_bytree': 0.3, 'min_child_samples': 95, 'n_estimators': 32, 'num_leaves': 20}

    Best hyperparameters preop:
 
    XGBoost:  {'colsample_bytree': 0.45, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 32}
    LightGBM: {'colsample_bytree': 0.2, 'min_child_samples': 95, 'n_estimators': 64, 'num_leaves': 20}

'''

def train_clf_new(x_dev, y_dev, ml_model = 'XGB'):

    # x_dev, x_test, y_dev, y_test = split_scale (x, y, scale = 'None')
    x_train,x_val,y_train,y_val = train_test_split(x_dev,y_dev,test_size = 0.2, random_state = 10, shuffle = True, stratify = y_dev)    

    if ml_model == 'XGB':
        clf = xgb.XGBClassifier(subsample=0.5, random_state = 20, n_jobs = -1, gamma = 0.1, eta = 0.1, objective = 'binary:logistic',
                                colsample_bytree= 0.45, max_depth=5, min_child_weight=10, n_estimators = 32)
                            
        clf.fit(x_train, np.ravel(y_train), 
                eval_set = [(x_train, np.ravel(y_train)), (x_val, np.ravel(y_val))], 
                early_stopping_rounds = 5,
                eval_metric='logloss') #48 rounds
    
        pred_score = clf.predict_proba(x_val, ntree_limit=clf.best_ntree_limit)[:,1]
        
        # y_test_score = clf.predict_proba(x_test, ntree_limit=clf.best_ntree_limit)[:,1]
    
        # xgb.plot_importance(clf)
        
    else: 
        clf = lgb.LGBMClassifier(subsample=0.5, objective  = 'binary', random_state = 20, n_jobs = -1, learning_rate = 0.1,
                                 verbose = 1,colsample_bytree = 0.2, min_child_samples = 95, n_estimators=64, num_leaves=20)
                                 
   
        clf.fit(x_train, np.ravel(y_train), 
                eval_set = [(x_train, np.ravel(y_train)), (x_val, np.ravel(y_val))], 
                early_stopping_rounds = 5,
                eval_metric='logloss') #48 rounds
    
        pred_score = clf.predict_proba(x_val)[:,1]
        
        # y_test_score = clf.predict_proba(x_test)[:,1]
        
    # explainer = shap.TreeExplainer(clf)
    # shap_values = explainer.shap_values(x_test)

    # shap.summary_plot(shap_values, x_test, plot_type="bar")    
    
      
    fpr, tpr, auc_thresholds = roc_curve(y_val, pred_score)
    
    balanced_idx = np.argmax(tpr - fpr)
    balanced_threshold = auc_thresholds[balanced_idx] 
    
    # spe = 1-fpr
    # prevalence = np.count_nonzero(y_val)/len(y_val) # should be around 920/(5673+920)
    # npv = spe*(1-prevalence)/((1-tpr)*prevalence + spe*(1-prevalence))

    # max_npv_idx = np.argmax(npv>=0.9)
    # max_npv_threshold = auc_thresholds[max_npv_idx] 
    
    return clf, balanced_threshold


ml_type = 'XGB'

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12345)

thresh_list, acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list= [], [], [], [], [], [], []
tn_list, fp_list, fn_list, tp_list = [], [], [] ,[]
fold_list = []
df_dict = {}

feature_importance_list = []

fold_num = 1

for train_index, test_index in skf.split(x, y):
    
    x_dev, x_test = x.iloc[train_index], x.iloc[test_index]
    y_dev, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf, op_threshold = train_clf_new(x_dev.reset_index(drop = True), y_dev.reset_index(drop = True), ml_model = ml_type)
    # y_test_score = clf.predict_proba(x_test)[:,1]
    # accuracy, tpr, tnr, npv, roc_score, ap_score = get_metrics_testset(y_test.to_numpy(), y_test_score, op_threshold)
    
    y_test_score = clf.predict_proba(x_test.values)[:,1]   
    y_pred_lbl = [1 if y >= op_threshold else 0 for y in y_test_score]

    df_dict[fold_num] = pd.DataFrame(list(zip(test_index, y_test_score, y_pred_lbl, y_test.values.flatten())),
                         columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])
    
    tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap_score = get_metrics_testset(y_test.to_numpy(), y_test_score, op_threshold)
    
    fold_list.append(fold_num)
    thresh_list.append(op_threshold)
    acc_list.append(accuracy)
    tpr_list.append(tpr)
    tnr_list.append(tnr)
    pre_list.append(pre)
    auc_list.append(roc_score)
    ap_list.append(ap_score)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)
    
    feature_importance = permutation_importance(clf, x_test, y_test, scoring='roc_auc', n_repeats=50, random_state=10, n_jobs = -1) #scoring='neg_log_loss',

    result = pd.DataFrame(data = list(zip(fold_list, thresh_list, tn_list, fp_list, fn_list, tp_list, 
                                          acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list)),
                          columns= ['Fold', 'Threshold', 'TN', 'FP', 'FN', 'TP',
                                    'Accuracy', 'Recall', 'Specificity', 'Precision', 'ROC', 'AP'])

    feature_importance_list.append(feature_importance.importances_mean)
    
    fold_num +=1
    
df_result = pd.concat(df_dict).droplevel(1, axis=0).reset_index()  
df_result.rename(columns={'index': 'Fold'}, inplace=True) 
 
a = np.vstack(feature_importance_list)

df_importance = pd.DataFrame(list(zip(np.mean(a, axis = 0), np.std(a, axis = 0)/np.sqrt(len(a)))),
                             index=list(x_test.columns),
                             columns = ['Mean', 'SEM'])

df_result.to_excel(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), sheet_name='case', index = False)
with pd.ExcelWriter(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), engine='openpyxl', mode='a') as writer:  
    # df_result.to_excel(writer, sheet_name='case', index = False)
    result.to_excel(writer, sheet_name='sum', index=False)
    df_importance.to_excel(writer, sheet_name='importance', index=True) 
    
plot_feature_importance(df_importance, top = 20, title_text = target[0].upper())







''' plot importance score from excel'''

# df_rename = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\PAD\Variable names.xlsx')


ml_type = 'LR'

df_importance = pd.read_csv(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\Esophageal cancer lymph node\Results\Neck LN\Permutation score {}-AUC.csv'.format(ml_type), index_col=0)  

# a = dict(df_rename.values)

# b = df_importance.rename(index=a)

plot_feature_importance(df_importance, top = 20, title_text = 'Neck-LN')





''' Baseline model '''
import math

def row_operation(row):
    img_finding = 1 if row['Diameter'] > 0 else 0
    t = row['Size'] * 0.378 + row['Chest']*1.371 + row['RLN']*1.340 + img_finding*1.488 - 3.945 
    p = 1 / (1 + math.exp(-t))
    return p

def bsl_model(x_dev, y_dev):
    # p = 1 / (1 + e -(-3.945 + 0.378 * Diameter + 1.371 * PLN + 1.340 * RLN + 1.488 * CT))
     
    select_feature = ['Size', 'Chest', 'RLN', 'Diameter']
    target = ['Neck_LN']
    
    # beta0 = -3.945
    # betas = [0.378, 1.371, 1.340, 1.488] # double check the order
    
    # betaxbar = sum(a*b for a,b in zip(jama_model_coef,jama_model_xbar))
    
    x_train = x_dev.loc[:, select_feature]
    y_train = y_dev.loc[:, target].values
    
    y_pred = x_train.apply(row_operation, axis=1).to_numpy()
    
    fpr, tpr, auc_thresholds = roc_curve(y_train, y_pred)
    balanced_idx = np.argmax(tpr - fpr)
    balanced_threshold = auc_thresholds[balanced_idx] 
    return balanced_threshold


ml_type = 'BSL'

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12345)

thresh_list, acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list= [], [], [], [], [], [], []
tn_list, fp_list, fn_list, tp_list = [], [], [] ,[]
fold_list = []
df_dict = {}

feature_importance_list = []

fold_num = 1

for train_index, test_index in skf.split(x, y):
    
    x_dev, x_test = x.iloc[train_index], x.iloc[test_index]
    y_dev, y_test = y.iloc[train_index], y.iloc[test_index]
    
    op_threshold = bsl_model(x_dev.reset_index(drop = True), y_dev.reset_index(drop = True))
    
    y_test_score = x_test.apply(row_operation, axis=1).to_numpy()
    

    y_pred_lbl = [1 if y >= op_threshold else 0 for y in y_test_score]

    df_dict[fold_num] = pd.DataFrame(list(zip(test_index, y_test_score, y_pred_lbl, y_test.values.flatten())),
                         columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])
    
    tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap_score = get_metrics_testset(y_test.to_numpy(), y_test_score, op_threshold)
    
    fold_list.append(fold_num)
    thresh_list.append(op_threshold)
    acc_list.append(accuracy)
    tpr_list.append(tpr)
    tnr_list.append(tnr)
    pre_list.append(pre)
    auc_list.append(roc_score)
    ap_list.append(ap_score)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)
    
    # feature_importance = permutation_importance(clf, x_test, y_test, scoring='roc_auc', n_repeats=50, random_state=10, n_jobs = -1) #scoring='neg_log_loss',

    result = pd.DataFrame(data = list(zip(fold_list, thresh_list, tn_list, fp_list, fn_list, tp_list, 
                                          acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list)),
                          columns= ['Fold', 'Threshold', 'TN', 'FP', 'FN', 'TP',
                                    'Accuracy', 'Recall', 'Specificity', 'Precision', 'ROC', 'AP'])

    # feature_importance_list.append(feature_importance.importances_mean)
    
    fold_num +=1

    # feature_importance_list.append(feature_importance.importances_mean)
    
    
df_result = pd.concat(df_dict).droplevel(1, axis=0).reset_index()  
df_result.rename(columns={'index': 'Fold'}, inplace=True) 
 
# a = np.vstack(feature_importance_list)

# df_importance = pd.DataFrame(list(zip(np.mean(a, axis = 0), np.std(a, axis = 0)/np.sqrt(len(a)))),
#                              index=list(x_test.columns),
#                              columns = ['Mean', 'SEM'])

df_result.to_excel(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), sheet_name='case', index = False)
with pd.ExcelWriter(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), engine='openpyxl', mode='a') as writer:  
    # df_result.to_excel(writer, sheet_name='case', index = False)
    result.to_excel(writer, sheet_name='sum', index=False)
    # df_importance.to_excel(writer, sheet_name='importance', index=True) 
    
# plot_feature_importance(df_importance, top = 20, title_text = target[0].upper())

''' preop
Accuracy	Recall	Specificity	Precision	ROC	AP

RF: 0.68129028	0.648148148	0.696316527	0.496221532	0.722907459	0.524468931
XGB: 0.668553584	0.640740741	0.68105042	0.480287293	0.711225879	0.528883457
LGB: 0.677808783	0.611111111	0.707871148	0.4915176	0.71467411	0.547050332


all
RF: 0.628144309	0.692592593	0.598809524	0.45930592	0.731976216	0.572322596
XGB: 0.681250415	0.6	0.717815126	0.502365796	0.73208476	0.565592979
LGB: 0.667444024	0.592592593	0.701162465	0.477644677	0.700913477	0.535769789

Baseline-nomo: 0.629386752	0.592592593	0.645840336	0.454206776	0.68424279	0.498029272
Baseline-ultrasound: 0.698491795	0.485185185	0.794607843	0.516396895	0.639896514	0.4116685


'''

'''Significance: Cochran's Q test for identical binomial proportions'''



ml_type = 'RF'

df_preop = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Preop only\{}.xlsx'.format(ml_type), 
                         sheet_name = 'case', header = 0)

df_all = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\All features\{}.xlsx'.format(ml_type), 
                         sheet_name = 'case', header = 0)


# array_like, 2d (N, k)
df_final = df_preop.merge(df_all, on =['ID','Fold'] , suffixes=('_preop', '_all'), how = 'outer')

df_final['sucess_preop'] = 1 - abs(df_final['Pred lbl_preop'] - df_final['Truth_preop'])
df_final['sucess_all'] = 1 - abs(df_final['Pred lbl_all'] - df_final['Truth_all'])

print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('All: {:.2%}'.format(len(df_final.loc[df_final['sucess_all'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_all']])
print(res)

''' only positive'''

df_final = df_final.loc[df_final['Truth_preop'] == 1].reset_index(drop=True)
print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('All: {:.2%}'.format(len(df_final.loc[df_final['sucess_all'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_all']])
print(res)



''' all non-sig except for RF preop even better'''

ml_type = 'RF'

df_preop = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Preop only\{}.xlsx'.format(ml_type), 
                         sheet_name = 'case', header = 0)

df_bsl = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Baseline\BSL-nomo.xlsx', 
                         sheet_name = 'case', header = 0)
# array_like, 2d (N, k)
df_final = df_preop.merge(df_bsl, on =['ID','Fold'] , suffixes=('_preop', '_bsl'), how = 'outer')

df_final['sucess_preop'] = 1 - abs(df_final['Pred lbl_preop'] - df_final['Truth_preop'])
df_final['sucess_bsl'] = 1 - abs(df_final['Pred lbl_bsl'] - df_final['Truth_bsl'])

print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_bsl']])
print(res)

''' only positive'''

df_final = df_final.loc[df_final['Truth_preop'] == 1].reset_index(drop=True)
print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_bsl']])
print(res)


''' ultrasound only'''

ml_type = 'BSL-Ult'

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12345)

thresh_list, acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list= [], [], [], [], [], [], []
tn_list, fp_list, fn_list, tp_list = [], [], [] ,[]
fold_list = []
df_dict = {}

fold_num = 1

for train_index, test_index in skf.split(x, y):
    
    x_dev, x_test = x.iloc[train_index], x.iloc[test_index]
    y_dev, y_test = y.iloc[train_index], y.iloc[test_index]
    
    y_test_score = (x_test['Diameter']>0).astype(int).to_numpy()
    op_threshold = 0.5

    y_pred_lbl = (x_test['Diameter']>0).astype(int).to_numpy()

    df_dict[fold_num] = pd.DataFrame(list(zip(test_index, y_test_score, y_pred_lbl, y_test.values.flatten())),
                             columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])
    
    tn, fp, fn, tp, accuracy, tpr, tnr, pre, roc_score, ap_score = get_metrics_testset(y_test.to_numpy(), y_test_score, op_threshold)
    
    fold_list.append(fold_num)
    thresh_list.append(op_threshold)
    acc_list.append(accuracy)
    tpr_list.append(tpr)
    tnr_list.append(tnr)
    pre_list.append(pre)
    auc_list.append(roc_score)
    ap_list.append(ap_score)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)
    
    result = pd.DataFrame(data = list(zip(fold_list, thresh_list, tn_list, fp_list, fn_list, tp_list, 
                                          acc_list, tpr_list, tnr_list, pre_list, auc_list, ap_list)),
                          columns= ['Fold', 'Threshold', 'TN', 'FP', 'FN', 'TP',
                                    'Accuracy', 'Recall', 'Specificity', 'Precision', 'ROC', 'AP'])

    fold_num +=1
   
    
df_result = pd.concat(df_dict).droplevel(1, axis=0).reset_index()  
df_result.rename(columns={'index': 'Fold'}, inplace=True) 

df_result.to_excel(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), sheet_name='case', index = False)
with pd.ExcelWriter(r'C:\Users\liy45\Desktop\{}.xlsx'.format(ml_type), engine='openpyxl', mode='a') as writer:  
    # df_result.to_excel(writer, sheet_name='case', index = False)
    result.to_excel(writer, sheet_name='sum', index=False)




''' RF-preop vs baseline-ultrasound'''

ml_type = 'RF'

df_preop = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Preop only\{}.xlsx'.format(ml_type), 
                         sheet_name = 'case', header = 0)

df_bsl = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Baseline\BSL-Ult.xlsx', 
                         sheet_name = 'case', header = 0)
# array_like, 2d (N, k)
df_final = df_preop.merge(df_bsl, on =['ID','Fold'] , suffixes=('_preop', '_bsl'), how = 'outer')

df_final['sucess_preop'] = 1 - abs(df_final['Pred lbl_preop'] - df_final['Truth_preop'])
df_final['sucess_bsl'] = 1 - abs(df_final['Pred lbl_bsl'] - df_final['Truth_bsl'])

print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_bsl']])
print(res)




''' only positive'''

df_final = df_final.loc[df_final['Truth_preop'] == 1].reset_index(drop=True)
print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_bsl']])
print(res)


''' ultrasound vs nomogram'''

df_preop = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Baseline\BSL-Ult.xlsx', 
                         sheet_name = 'case', header = 0)

df_bsl = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Baseline\BSL-nomo.xlsx', 
                         sheet_name = 'case', header = 0)
# array_like, 2d (N, k)
df_final = df_preop.merge(df_bsl, on =['ID','Fold'] , suffixes=('_preop', '_bsl'), how = 'outer')

df_final['sucess_preop'] = 1 - abs(df_final['Pred lbl_preop'] - df_final['Truth_preop'])
df_final['sucess_bsl'] = 1 - abs(df_final['Pred lbl_bsl'] - df_final['Truth_bsl'])

print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_bsl']])
print(res)

''' only positive'''

df_final = df_final.loc[df_final['Truth_preop'] == 1].reset_index(drop=True)
print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_bsl']])
print(res)



''' between preop models'''

ml_type = 'SVM'

df_preop = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Preop only\{}.xlsx'.format(ml_type), 
                         sheet_name = 'case', header = 0)

df_bsl = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Baseline\BSL-nomo.xlsx', 
                         sheet_name = 'case', header = 0)
# array_like, 2d (N, k)
df_final = df_preop.merge(df_bsl, on =['ID','Fold'] , suffixes=('_preop', '_bsl'), how = 'outer')

df_final['sucess_preop'] = 1 - abs(df_final['Pred lbl_preop'] - df_final['Truth_preop'])
df_final['sucess_bsl'] = 1 - abs(df_final['Pred lbl_bsl'] - df_final['Truth_bsl'])

print('Preop: {:.2%}'.format(len(df_final.loc[df_final['sucess_preop'] == 1])/len(df_final)))
print('Baseline: {:.2%}'.format(len(df_final.loc[df_final['sucess_bsl'] == 1])/len(df_final)))
res = cochrans_q(df_final[['sucess_preop', 'sucess_bsl']])
print(res)
