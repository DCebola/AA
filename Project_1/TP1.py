#TODO: Three Classifiers:
# Logistic Regression
# find the best value for the regularization parameter C
# C is in [1e-2, 1e12] each step is *10 (1e-2 ,1e-1 ,1e0,...)
# plot of cross-validation and training errors for the C parameter
# ---------------------------
# Custom Naive Bayes classifier using Kernel Density Estimation
# Kernel Density Estimation for the probability distributions of the feature values
# Use the KernelDensity class from sklearn.neighbors.kde for the density estimation
# KDE is in [0.02,0.6] each step is +0.02
# Accuracy is measured by accuracy_score (sklearn.metrics.accuracy_score)
# find the optimum value for the bandwitdh parameter of the kernel density estimators
# plot of training and cross-validation errors for the KDE kernel
# ---------------------------
# Gaussian Naive Bayes classifier in the sklearn.naive_bayes.GaussianNB class
# ---------------------------
# Cross Validation each classifier with 5 folds
# Compare all classifiers, identify the best one and discuss if it is significantly better than the others
# For comparing the classifiers, use the approximate normal test and McNemar's test, both with a 95% confidence interval


import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

FOLDS = 5
KDE_STEP = 0.02
MAX_C_EXPONENT = 12
MIN_C_EXPONENT = -2
MIN_KDE = 0.02
MAX_KDE = 0.6
FEATS = 4


def standardize(_train_data, _test_data):
    _train_features = _train_data[:, 0:4]
    _train_classes = _train_data[:, 4]
    _test_features = _test_data[:, 0:4]
    _test_classes = _test_data[:, 4]
    f_means = np.mean(_train_features)
    f_std = np.std(_train_features)
    _train_features = (_train_features - f_means) / f_std
    _test_features = (_test_features - f_means) / f_std
    return np.column_stack((_train_features, _train_classes)), np.column_stack((_test_features, _test_classes))


def calc_fold(X, Y, train_ix, valid_ix, C):
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix, :FEATS], Y[train_ix])
    prob = reg.predict_proba(X[:, :FEATS])[:, 1]
    squares = (prob - Y) ** 2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])


def logistic_regression(_train_data, ):
    best_val_err = 100000000
    kf = StratifiedKFold(n_splits=FOLDS)
    best_exp = -3
    for exp in range(MIN_C_EXPONENT, MAX_C_EXPONENT + 1):
        tr_err = va_err = 0
        for tr_ix, va_ix in kf.split(_train_data[:, 4], _train_data[:, 4]):
            fold_t_err, fold_v_err = calc_fold(_train_data[:, 0:4], _train_data[:, 4], tr_ix, va_ix, 10 ** exp)
            tr_err += fold_t_err
            va_err += fold_v_err
        if va_err / FOLDS < best_val_err:
            best_val_err = va_err / FOLDS
            best_exp = exp
    return 10**best_exp


def custom_naive_bayes(kde, _train_data):
    pass


np.set_printoptions(precision=4)
test_data = np.loadtxt('TP1_test.tsv')
train_data = np.loadtxt('TP1_train.tsv')
train_data, test_data = standardize(shuffle(train_data), shuffle(test_data))

c = logistic_regression(train_data)
