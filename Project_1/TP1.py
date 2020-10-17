# TODO: Three Classifiers:
#  Logistic Regression
#  find the best value for the regularization parameter C
#  C is in [1e-2, 1e12] each step is *10 (1e-2 ,1e-1 ,1e0,...)
#  plot of cross-validation and training errors for the C parameter
#  ---------------------------
#  Custom Naive Bayes classifier using Kernel Density Estimation
#  Kernel Density Estimation for the probability distributions of the feature values
#  Use the KernelDensity class from sklearn.neighbors.kde for the density estimation
#  KDE is in [0.02,0.6] each step is +0.02
#  Accuracy is measured by accuracy_score (sklearn.metrics.accuracy_score)
#  find the optimum value for the bandwitdh parameter of the kernel density estimators
#  plot of training and cross-validation errors for the KDE kernel
#  ---------------------------
#  Gaussian Naive Bayes classifier in the sklearn.naive_bayes.GaussianNB class
#  ---------------------------
#  Cross Validation each classifier with 5 folds
#  Compare all classifiers, identify the best one and discuss if it is significantly better than the others
#  For comparing the classifiers, use the approximate normal test and McNemar's test, both with a 95% confidence interval


import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB

FOLDS = 5
MAX_C_EXPONENT = 12
MIN_C_EXPONENT = -2
KDE_STEP = 2
MIN_KDE = 2
MAX_KDE = 60
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


def calc_fold_logistic(x, y, train_ix, valid_ix, _c):
    reg = LogisticRegression(C=_c, tol=1e-10)
    reg.fit(x[train_ix], y[train_ix])
    squares = (reg.predict_proba(x[:, :FEATS])[:, 1] - y) ** 2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])


def logistic_regression(_train_data, kf):
    best_val_err = 100000000
    best_exp = -3
    for exp in range(MIN_C_EXPONENT, MAX_C_EXPONENT + 1):
        tr_err = va_err = 0
        for tr_ix, va_ix in kf.split(_train_data[:, 4], _train_data[:, 4]):
            fold_t_err, fold_v_err = calc_fold_logistic(_train_data[:, 0:4], _train_data[:, 4], tr_ix, va_ix, 10 ** exp)
            tr_err += fold_t_err
            va_err += fold_v_err
        if va_err / FOLDS < best_val_err:
            best_val_err = va_err / FOLDS
            best_exp = exp
    return 10 ** best_exp


def calc_fold_bayes(x, y, train_ix, valid_ix, __h):
    kde_1 = KernelDensity(bandwidth=__h)
    kde_0 = KernelDensity(bandwidth=__h)
    # for the fold selected points we need to train the KDE for each class,  meaning for all points that belong to
    # one class are needed to fit kde.fit([points of class 1]) and kde.fit([points fo class 2])
    #
    class1_train_points, class0_train_points = separate_classes(x, y, train_ix)

    class1_valid_test_points, class0_valid_test_points = separate_classes(x, y, valid_ix)
    kde_0.fit(class0_train_points)
    kde_1.fit(class1_train_points)
    print("x[train_ix]")
    print(x[train_ix])
    print("class0")
    print(class0_train_points)
    logs_class0 = kde_0.score_samples(class0_valid_test_points)
    class0_log = len(class0_valid_test_points[0 ])/len(valid_ix)
    logs_class1 = kde_1.score_samples(class1_valid_test_points)


    return 0, 0


def separate_classes(x, y, indexes):
    class1 = np.zeros((int(sum(y[indexes])), x[indexes].shape[1]))
    class0 = np.zeros((len(y[indexes])-int(sum(y[indexes])), x[indexes].shape[1]))
    print(class1.shape)
    print(class0.shape)
    counter = 0
    indexed_x = x[indexes]
    indexed_y = y[indexes]
    for value in indexed_y:
        if value == 0:
            class0[counter, :] = class0[counter, :] + indexed_x[counter]
        else:
            class1[counter, :] = class1[counter, :] + indexed_x[counter]
        counter = counter + 1
    return class0, class1


def custom_naive_bayes(_train_data, kf):
    best_val_err = 100000000
    best_h = -1
    h_values = map(lambda x: x / 100.0, range(MIN_KDE, MAX_KDE, KDE_STEP))
    for _h in h_values:
        tr_err = va_err = 0
        for tr_ix, va_ix in kf.split(_train_data[:, 4], _train_data[:, 4]):
            fold_t_err, fold_v_err = calc_fold_bayes(_train_data[:, 0:4], _train_data[:, 4], tr_ix, va_ix, _h)
            tr_err += fold_t_err
            va_err += fold_v_err
        if va_err / FOLDS < best_val_err:
            best_val_err = va_err / FOLDS
            best_h = _h
    return best_h


np.set_printoptions(precision=4)
test_data = np.loadtxt('TP1_test.tsv')
train_data = np.loadtxt('TP1_train.tsv')
train_data, test_data = standardize(shuffle(train_data), shuffle(test_data))

c = logistic_regression(train_data, StratifiedKFold(n_splits=FOLDS))
h = custom_naive_bayes(train_data, StratifiedKFold(n_splits=FOLDS))
