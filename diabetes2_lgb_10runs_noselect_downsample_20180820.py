# -*- coding: utf-8 -*-
"""GITdiabetes2_lgb-10runs-noselect-downsample-20180820.ipynb
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing

import scipy.io
import time
import matplotlib.pyplot as plt

from scipy import stats
import copy

# d = { ... }
# d2 = copy.deepcopy(d)

tic = time.time()

name = pd.read_csv("name.csv")
name = name.iloc[0]  # name.T
print(name.values.shape)

mat = scipy.io.loadmat("data_26478_1286.mat")
# X=mat['extended_allFeas']
y = mat["labels"] - 1
df_new = pd.DataFrame(mat["extended_allFeas"], columns=name.values)
# df_new.describe()

df_new["y"] = y

df = df_new.loc[:, df_new.std() > 0.01].copy()
print(df.shape)


name = df.columns.values


def balanced_subsample(x, y, subsample_size=1.0):
    np.random.seed(2018)
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys


"""# lgb

https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

To get good results using a leaf-wise tree, these are some important parameters:

    num_leaves. This is the main parameter to control the complexity of the tree model. Theoretically, we can set num_leaves = 2^(max_depth) to obtain the same number of leaves as depth-wise tree. However, this simple conversion is not good in practice. The reason is that a leaf-wise tree is typically much deeper than a depth-wise tree for a fixed number of leaves. Unconstrained depth can induce over-fitting. Thus, when trying to tune the num_leaves, we should let it be smaller than 2^(max_depth). For example, when the max_depth=7 the depth-wise tree can get good accuracy, but setting num_leaves to 127 may cause over-fitting, and setting it to 70 or 80 may get better accuracy than depth-wise.
    min_data_in_leaf. This is a very important parameter to prevent over-fitting in a leaf-wise tree. Its optimal value depends on the number of training samples and num_leaves. Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to hundreds or thousands is enough for a large dataset.
    max_depth. You also can use max_depth to limit the tree depth explicitly.

For Faster Speed

    Use bagging by setting bagging_fraction and bagging_freq
    Use feature sub-sampling by setting feature_fraction
    Use small max_bin
    Use save_binary to speed up data loading in future learning
    Use parallel learning, refer to Parallel Learning Guide

For Better Accuracy

    Use large max_bin (may be slower)
    Use small learning_rate with large num_iterations
    Use large num_leaves (may cause over-fitting)
    Use bigger training data
    Try dart

Deal with Over-fitting

    Use small max_bin
    Use small num_leaves
    Use min_data_in_leaf and min_sum_hessian_in_leaf
    Use bagging by set bagging_fraction and bagging_freq
    Use feature sub-sampling by set feature_fraction
    Use bigger training data
    Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
    Try max_depth to avoid growing deep tree


"""

score10 = []
num_round = 1000

ytest10 = np.zeros((round(df.shape[0] * 0.6), 10))  # pd.DataFrame()
lgb_preds10 = np.zeros((round(df.shape[0] * 0.6), 10))  # pd.DataFrame()

for seed in np.array(range(0, 10, 1)):
    X = df.drop("y", axis=1).values
    X_train, xtest, y_train, ytest = train_test_split(
        X, y, stratify=y, test_size=0.6, random_state=seed
    )
    xtrain, xvalid, ytrain, yvalid = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.3, random_state=seed
    )
    xtrain, ytrain = balanced_subsample(xtrain, ytrain.ravel(), subsample_size=1.0)
    xvalid, yvalid = balanced_subsample(xvalid, yvalid.ravel(), subsample_size=1.0)
    print(xtrain.shape, xvalid.shape)
    dtrain = lgb.Dataset(
        xtrain, ytrain.ravel(), free_raw_data=False
    )  # lgb.Dataset(X,  y, free_raw_data=False)
    dval = lgb.Dataset(xvalid, yvalid.ravel(), reference=dtrain, free_raw_data=False)

    best_score = 0.5
    for i in np.array(range(80, 141, 10)):  #
        np.random.seed(i)  # +1000
        params = {
            "learning_rate": [0.05],
            "boosting_type": ["gbdt"],
            "objective": ["binary"],  #'regression',#
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "metric": "auc",  #'rmse',#'RMSE',#'rmsle',
            "seed": np.random.randint(1, 10000),  # i,
            "silent": True,  # ,
        }
        params["nthread"] = 9  # 16#28#16#
        params["num_leaves"] = i  #

        bst = lgb.train(
            params,
            dtrain,
            num_round,
            valid_sets=dval,
            verbose_eval=500,
            early_stopping_rounds=50,
            feature_name=list(df.drop("y", axis=1).columns),
        )  # , feval=rmsle  valid_sets=[dtrain, dval], ,categorical_feature=list(cat_col)
        lgb_preds = bst.predict(xvalid, num_iteration=bst.best_iteration)
        fpr, tpr, thresholds = roc_curve(yvalid, lgb_preds)
        score1 = auc(fpr, tpr)
        if score1 > best_score:  # 0.9:#
            best_score = score1
            best_params = copy.deepcopy(params)
            best_nround = bst.best_iteration  #

    print(best_score, best_nround)
    print(best_params["num_leaves"])
    ##########################################
    params = copy.deepcopy(best_params)  #
    best_score = 0.5
    for j in np.array(range(14, 26, 2)):
        np.random.seed(j)  # +1000
        params["min_data_in_leaf"] = j
        bst = lgb.train(
            params,
            dtrain,
            num_round,
            valid_sets=dval,
            verbose_eval=500,
            early_stopping_rounds=50,
            feature_name=list(df.drop("y", axis=1).columns),
        )  # , feval=rmsle  valid_sets=[dtrain, dval], ,categorical_feature=list(cat_col)
        lgb_preds = bst.predict(xvalid, num_iteration=bst.best_iteration)
        fpr, tpr, thresholds = roc_curve(yvalid, lgb_preds)
        score1 = auc(fpr, tpr)
        if score1 > best_score:  # 0.9:#
            best_score = score1
            best_params = copy.deepcopy(params)
            best_nround = bst.best_iteration  # score.iloc[:,0].idxmin()

    print(best_score, best_nround)
    print(best_params["min_data_in_leaf"])
    ##########################################
    params = copy.deepcopy(best_params)  #
    best_score = 0.5
    for j in np.array(range(1, 10, 1)):
        np.random.seed(j)  # +1000
        params["learparams =ning_rate"] = (0.01 * j,)
        bst = lgb.train(
            params,
            dtrain,
            num_round,
            valid_sets=dval,
            verbose_eval=500,
            early_stopping_rounds=50,
            feature_name=list(df.drop("y", axis=1).columns),
        )  # , feval=rmsle  valid_sets=[dtrain, dval], ,categorical_feature=list(cat_col)
        lgb_preds = bst.predict(xvalid, num_iteration=bst.best_iteration)
        fpr, tpr, thresholds = roc_curve(yvalid, lgb_preds)
        score1 = auc(fpr, tpr)
        if score1 > best_score:  # 0.9:#
            best_score = score1
            best_params = copy.deepcopy(params)
            best_nround = bst.best_iteration  # score.iloc[:,0].idxmin()

    print(best_score, best_nround)
    print(best_params)

    bst = lgb.train(
        best_params,
        lgb.Dataset(
            np.vstack((xtrain, xvalid)),
            np.hstack((ytrain.ravel(), yvalid.ravel())),
            free_raw_data=False,
        ),
        best_nround,
        valid_sets=dval,
        verbose_eval=500,
        early_stopping_rounds=50,
        feature_name=list(df.drop("y", axis=1).columns),
    )  # ,categorical_feature=list(cat_col)
    # ,categorical_feature=list(cat_col)
    lgb_preds = bst.predict(xtest, num_iteration=bst.best_iteration)
    ytest10[:, seed], lgb_preds10[:, seed] = ytest.ravel(), lgb_preds.ravel()
    fpr, tpr, thresholds = roc_curve(ytest, lgb_preds)
    score1 = auc(fpr, tpr)
    print(score1)
    score10.append(score1)

df_ytest10 = pd.DataFrame(ytest10)
df_lgb_preds10 = pd.DataFrame(lgb_preds10)
df_lgb_preds10.head()

df_ytest10.to_csv("df_ytest10", index=False)  # encoding='utf-8',
df_lgb_preds10.to_csv("df_lgb_preds10", index=False)

np.vstack((xtrain, xvalid)).shape, np.hstack((ytrain.ravel(), yvalid.ravel())).shape

y_train.mean(), ytrain.mean()

score10

scores10 = np.array(score10)

scores10.mean(), scores10.std()

fig, ax = plt.subplots(figsize=(10, 14))
lgb.plot_importance(bst, max_num_features=60, ax=ax)
plt.title("Light GBM Feature Importance")

time.time() - tic
