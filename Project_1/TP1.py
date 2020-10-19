# TODO: Three Classifiers:
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
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from math import sqrt

from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

FOLDS = 5
MAX_C_EXPONENT = 12
MIN_C_EXPONENT = -2
KDE_STEP = 2
MIN_KDE = 2
MAX_KDE = 60
FEATS = 4


def plot_errs(x, y_train, y_valid, title, x_title, filename):
    fig, ax = plt.subplots(figsize=(19, 10))

    ax.plot(x, y_train, '-r')
    ax.plot(x, y_valid, '-b')

    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.legend(['train_error', 'validation_error'])
    plt.savefig(filename)
    plt.show()
    plt.close(fig)


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


def logistic_regression(_train_data, _test_data, _c):
    reg = LogisticRegression(C=_c, tol=1e-10)
    reg.fit(_train_data[:, 0:FEATS], _train_data[:, FEATS])
    pred = reg.predict(_test_data[:, 0:FEATS])
    return 1 - reg.score(_train_data[:, 0:FEATS], _train_data[:, FEATS]), \
           1 - reg.score(_test_data[:, 0:FEATS], _test_data[:, FEATS]), \
           _test_data.shape[0] - get_score(pred, _test_data[:, FEATS], False), \
           pred


def get_score(_predicted, _test_data, n=False):
    return accuracy_score(_predicted, _test_data, normalize=n)


def custom_naive_bayes(_train_data, _test_data, _h):
    def separate_classes(feats, Ys):
        class1 = feats[np.where(Ys == 1)]
        class0 = feats[np.where(Ys == 0)]
        return class0, class1

    def calc_distrib(x_train, x_test):
        kde = KernelDensity(bandwidth=_h)
        kde.fit(np.reshape(x_train, (-1, 1)))
        return kde.score_samples(np.reshape(x_test, (-1, 1)))

    def create_distrib_matrix(feats_train, feats_test):
        log_features = np.zeros(feats_test.shape)
        for feat in range(FEATS):
            log_features[:, feat] = calc_distrib(feats_train[:, feat], feats_test[:, feat])
        return log_features

    def classify(features_class0, _log_class0, features_class1, _log_class1, _test_set):
        classes = np.zeros(_test_set.shape[0],)
        for i in range(_test_set.shape[0]):
            class0_sum = _log_class0
            class1_sum = _log_class1
            for feat in range(FEATS):
                class0_sum = class0_sum + features_class0[i, :][feat]
                class1_sum = class1_sum + features_class1[i, :][feat]
            if class0_sum < class1_sum:
                classes[i] = 1
        return classes

    def custom_naive_bayes_score(data_set, log_features_class0, log_features_class1, class0_log, class1_log):
        _predicted = classify(log_features_class0, class0_log, log_features_class1, class1_log, data_set)
        return data_set.shape[0] - get_score(_predicted, data_set[:, FEATS], False), 1 - get_score(_predicted, data_set[:, FEATS], True), _predicted

    class0_train_points, class1_train_points = separate_classes(_train_data[:, 0:FEATS], _train_data[:, FEATS])

    train_log_features_class0 = create_distrib_matrix(class0_train_points, _train_data[:, 0:FEATS])
    train_log_features_class1 = create_distrib_matrix(class1_train_points, _train_data[:, 0:FEATS])

    test_log_features_class0 = create_distrib_matrix(class0_train_points, _test_data[:, 0:FEATS])
    test_log_features_class1 = create_distrib_matrix(class1_train_points, _test_data[:, 0:FEATS])

    total = _train_data.shape[0]
    log_class0 = np.log(float(class0_train_points.shape[0]) / total)
    log_class1 = np.log(float(class1_train_points.shape[0]) / total)
    train_errors, train_error_percentage = \
        custom_naive_bayes_score(_train_data, train_log_features_class0, train_log_features_class1, log_class0, log_class1)[0:2]

    test_errors, test_error_percentage, predicted = \
        custom_naive_bayes_score(_test_data, test_log_features_class0, test_log_features_class1, log_class0, log_class1)
    return train_error_percentage, test_error_percentage, test_errors, predicted


def cross_validation(_train_data, p_values, clf_function, kf):
    best_val_err = 100000000
    best_p = -1
    tr_errs = np.zeros((len(p_values),))
    va_errs = np.zeros((len(p_values),))
    counter = 0
    for p in tqdm(p_values, ncols=100, desc=clf_function.__name__):
        va_err = 0
        tr_err = 0
        for tr_ix, va_ix in kf.split(_train_data[:, 4], _train_data[:, 4]):
            fold_train_err, fold_va_err = clf_function(_train_data[tr_ix], _train_data[va_ix], p)[0:2]
            va_err += fold_va_err
            tr_err += fold_train_err
        tr_errs[counter] = tr_err / FOLDS
        va_errs[counter] = va_err / FOLDS
        counter += 1
        if va_err / FOLDS < best_val_err:
            best_val_err = va_err / FOLDS
            best_p = p
    return best_p, tr_errs, va_errs


def gaussian_naive_bayes(_train_data, _test_data):
    clf = GaussianNB()
    clf.fit(_train_data[:, 0:4], _train_data[:, 4])
    predicted = clf.predict(_test_data[:, 0:4])
    return _test_data.shape[0] - get_score(predicted, _test_data[:, 4], False), predicted


def approximate_normal_test(n, err):
    theta = sqrt(err * (1 - err / n))
    return err, 1.96 * theta


def mc_nemars_test(_test_data, clf_predicted_1, clf_predicted_2):
    e10 = 0
    e01 = 0
    for i in range(_test_data.shape[0]):
        if clf_predicted_1[i] != clf_predicted_2[i]:
            if _test_data[:, FEATS][i] == clf_predicted_1[i]:
                e01 += 1
            else:
                e10 += 1
    return ((abs(e01 - e10) - 1) ** 2) / (e01 + e10)


np.set_printoptions(precision=4)
test_data = np.loadtxt('TP1_test.tsv')
train_data = np.loadtxt('TP1_train.tsv')
train_data, test_data = standardize(shuffle(train_data), shuffle(test_data))
folds = StratifiedKFold(n_splits=FOLDS)
c_values = list(map(lambda x: 10 ** x, range(MIN_C_EXPONENT, MAX_C_EXPONENT + 1)))
h_values = list(map(lambda x: x / 100, range(MIN_KDE, MAX_KDE, KDE_STEP)))

c, c_tr_errs, c_va_errs = cross_validation(train_data, c_values, logistic_regression, folds)
h, h_tr_errs, h_va_errs = cross_validation(train_data, h_values, custom_naive_bayes, folds)

plot_errs(np.log10(c_values), c_tr_errs, c_va_errs, 'Logistic Regression', ' c value as 10 to the power of', "LR.png")
plot_errs(h_values, h_tr_errs, h_va_errs, 'KDE based Naive Bayes', 'h value: bandwidth', "NB.png")

sample_size = train_data.shape[0]
l_err, l_pred = logistic_regression(train_data, test_data, c)[2:4]
c_naive_bayes_err, c_naive_bayes_pred = custom_naive_bayes(train_data, test_data, h)[2:4]
naive_bayes_err, naive_bayes_pred = gaussian_naive_bayes(train_data, test_data)

print("----------------------\033[1mCrossValidation\033[0m----------------------")
print("C found: \033[1m" + str(c) + '\033[0m')
print("H found: \033[1m" + str(h) + '\033[0m')
print("--------------------------\033[1mScores\033[0m---------------------------")
print("Logistic Regression errors: \033[1m{err:.2f}\033[0m".format(err=l_err))
print("Custom gaussian naive bayes errors: \033[1m{err:.2f}\033[0m".format(err=c_naive_bayes_err))
print("Gaussian naive bayes errors: \033[1m{err:.2f}\033[0m".format(err=naive_bayes_err))
print("----------------\033[1mApproximate Normal Test (95%)\033[0m--------------")
print("Logistic Regression: \033[1m{0[0]:0.2f} ± {0[1]:0.2f}\033[0m".format(approximate_normal_test(sample_size, l_err)))
print("Custom gaussian naive bayes: \033[1m{0[0]:0.2f} ± {0[1]:0.2f}\033[0m".format(approximate_normal_test(sample_size, c_naive_bayes_err)))
print("Gaussian naive bayes: \033[1m{0[0]:0.2f} ± {0[1]:0.2f}\033[0m".format(approximate_normal_test(sample_size, naive_bayes_err)))
print("----------------------\033[1mMcNemar's Test\033[0m-----------------------")
print("Logistic Regression vs Custom gaussian naive bayes: \033[1m{score:.2f}\033[0m". \
      format(score=mc_nemars_test(test_data, l_pred, c_naive_bayes_pred)))
print("Logistic Regression vs Gaussian naive bayes: \033[1m{score:.2f}\033[0m". \
      format(score=mc_nemars_test(test_data, l_pred, naive_bayes_pred)))
print("Custom gaussian naive bayes vs Gaussian naive bayes: \033[1m{score:.2f}\033[0m". \
      format(score=mc_nemars_test(test_data, c_naive_bayes_pred, naive_bayes_pred)))
print("-----------------------------------------------------------")
