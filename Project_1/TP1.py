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


def separate_classes(x, y, indexes):
    indexed_x = x[indexes]
    indexed_y = y[indexes]
    class1 = indexed_x[np.where(indexed_y == 1)]
    class0 = indexed_x[np.where(indexed_y == 0)]

    return class0, class1


def kde_score(fit_x, score_x, _h):
    kde = KernelDensity(bandwidth=_h)
    kde.fit(fit_x)
    return np.array(kde.score_samples(score_x))


def calc_feat_log_class(feat, class_tr, class_va, _h):
    class_tr_feat = np.array(class_tr[:, feat]).reshape(-1, 1)
    class_va_feat = np.array(class_va[:, feat]).reshape(-1, 1)

    return kde_score(class_tr_feat, class_va_feat, _h)


''' def classify(features_class0, log_class0, features_class1, log_class1, feat_mat):
    classes = np.zeros(feat_mat.shape[0])
    for point in range(feat_mat.shape[0]):
        class0_sum = log_class0 + features_class0[point]
        class1_sum = log_class1 + features_class1[point]
        if class0_sum < class1_sum:
            classes[point] = 1
    return classes'''


def classify(feat_logs_class0, class0_log, feat_logs_class1, class1_log, indexes):
    classes = np.zeros((len(indexes[0]),))

    counter = 0
    for row in indexes:
        class0_sum = class0_log
        class1_sum = class1_log
        for column in range(feat_logs_class0.shape[1]):
            class0_sum = class0_sum + feat_logs_class0[row, :][0, column]
            class1_sum = class1_sum + feat_logs_class1[row, :][0, column]
        if class0_sum < class1_sum:
            classes[counter] = 1
        counter = counter + 1
    return classes


def obtain_feature_log(x, y, train_ix, valid_ix, _h):

    class0_tr_feats, class1_tr_feats = separate_classes(x, y, train_ix)
    test_feats = x[valid_ix]

    feat_logs_class0 = np.zeros(test_feats.shape)
    feat_logs_class1 = np.zeros(test_feats.shape)

    # FEATURE 0

    feat_logs_class0[:, 0] = feat_logs_class0[:, 0] + calc_feat_log_class(0, class0_tr_feats, test_feats, _h)
    # ==========================================================================================
    #  CLASS 0 ABOVE || CLASS 1 BELOW
    # ==========================================================================================

    feat_logs_class1[:, 0] = feat_logs_class1[:, 0] + calc_feat_log_class(0, class1_tr_feats, test_feats, _h)
    # FEATURE 0

    # FEATURE 1
    feat_logs_class0[:, 1] = feat_logs_class0[:, 1] + calc_feat_log_class(1, class0_tr_feats, test_feats, _h)
    # ==========================================================================================
    #  CLASS 0 ABOVE || CLASS 1 BELOW
    # ==========================================================================================
    feat_logs_class1[:, 1] = feat_logs_class1[:, 1] + calc_feat_log_class(1, class1_tr_feats, test_feats, _h)
    # FEATURE 1

    # FEATURE 2
    feat_logs_class0[:, 2] = feat_logs_class0[:, 2] + calc_feat_log_class(2, class0_tr_feats, test_feats, _h)
    # ==========================================================================================
    #  CLASS 0 ABOVE || CLASS 1 BELOW
    # ==========================================================================================
    feat_logs_class1[:, 2] = feat_logs_class1[:, 2] + calc_feat_log_class(2, class1_tr_feats, test_feats, _h)
    # FEATURE 2

    # FEATURE 3
    feat_logs_class0[:, 3] = feat_logs_class0[:, 3] + calc_feat_log_class(3, class0_tr_feats, test_feats, _h)
    # ==========================================================================================
    #  CLASS 0 ABOVE || CLASS 1 BELOW
    # ==========================================================================================
    feat_logs_class1[:, 3] = feat_logs_class1[:, 3] + calc_feat_log_class(3, class1_tr_feats, test_feats, _h)
    # FEATURE 3

    return feat_logs_class0, feat_logs_class1


def calc_fold_logistic(x, y, train_ix, valid_ix, _c):
    reg = LogisticRegression(C=_c, tol=1e-10)
    reg.fit(x[train_ix], y[train_ix])
    squares = (reg.predict_proba(x[:, :FEATS])[:, 1] - y) ** 2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])


def calc_fold_bayes(x, y, train_ix, valid_ix, _h):

    '''
    kde_1 = KernelDensity(bandwidth=_h)
        kde_0 = KernelDensity(bandwidth=_h)

        class0_train_points, class1_train_points = separate_classes(x, y, train_ix)
        class0_valid_test_points, class1_valid_test_points = separate_classes(x, y, valid_ix)

        kde_0.fit(class0_train_points)
        kde_1.fit(class1_train_points)

        log_features_class0 = kde_0.score_samples(x[valid_ix])
        log_features_class1 = kde_1.score_samples(x[valid_ix])

        total_len = log_features_class1.shape[0] + log_features_class0.shape[0]
        class0_log = class0_valid_test_points.shape[0] / total_len
        class1_log = class1_valid_test_points.shape[0] / total_len

        classed_0 = classify(log_features_class0, class0_log, log_features_class1, class1_log, class0_valid_test_points)
        classed_1 = classify(log_features_class0, class0_log, log_features_class1, class1_log, class1_valid_test_points)

        errors = sum(classed_0) + sum(1 - classed_1)
        error_percent = float(errors) / (len(classed_0) + len(classed_1)) * 100
        return error_percent
    '''

    feat_logs_class0, feat_logs_class1 = obtain_feature_log(x, y, train_ix, valid_ix, _h)
    classed_0_val, classed_1_val = separate_classes(x, y, valid_ix)

    #print(feat_logs_class1.shape)
    #print(feat_logs_class0.shape)
    valid_feats = y[valid_ix]
    classed_0 = classify(feat_logs_class0, 0, feat_logs_class1, 0, np.where(valid_feats == 0))
    classed_1 = classify(feat_logs_class0, 0, feat_logs_class1, 0, np.where(valid_feats == 1))
    errors = sum(classed_0) + sum(1 - classed_1)
    error_percent = float(errors) / (len(classed_0) + len(classed_1)) * 100
    print(f'errors = {errors:.0f}')
    print(f'Error percent = {error_percent:.2f}')
    return error_percent


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


def custom_naive_bayes(_train_data, kf):
    best_val_err = 100000000
    best_h = -1
    h_values = map(lambda x: x / 100.0, range(MIN_KDE, MAX_KDE, KDE_STEP))
    for _h in h_values:
        va_err = 0
        for tr_ix, va_ix in kf.split(_train_data[:, 4], _train_data[:, 4]):
            fold_v_err = calc_fold_bayes(_train_data[:, 0:4], _train_data[:, 4], tr_ix, va_ix, _h)
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

print(c)
print(h)
