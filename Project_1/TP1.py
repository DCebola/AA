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

from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True, )

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
    squares_train = (reg.predict_proba(_train_data[:, :FEATS])[:, 1] - _train_data[:, FEATS]) ** 2
    squares_test = (reg.predict_proba(_test_data[:, :FEATS])[:, 1] - _test_data[:, FEATS]) ** 2
    return np.mean(squares_train), np.mean(squares_test)


def custom_naive_bayes(_train_data, _test_data, _h):
    def separate_classes(feats, Ys):
        class1 = feats[np.where(Ys == 1)]
        class0 = feats[np.where(Ys == 0)]

        return class0, class1

    def calc_distrib(x_train, x_test):
        kde = KernelDensity(bandwidth=_h)
        kde.fit(np.reshape(x_train, (-1, 1)))
        return kde.score_samples(np.reshape(x_test, (-1, 1)))

    def get_score(predicted, mask, n=False):
        return accuracy_score(predicted, mask, normalize=n)

    def create_distrib_matrix(feats_train, feats_test):
        log_features = np.zeros(feats_test.shape)
        for feat in range(FEATS):
            log_features[:, feat] = calc_distrib(feats_train[:, feat], feats_test[:, feat])
        return log_features

    def classify(features_class0, _log_class0, features_class1, _log_class1, indexes):
        classes = np.zeros((len(indexes[0]),))

        counter = 0
        for index in indexes[0]:
            class0_sum = _log_class0
            class1_sum = _log_class1
            for feat in range(FEATS):
                class0_sum = class0_sum + features_class0[index, :][feat]
                class1_sum = class1_sum + features_class1[index, :][feat]
            if class0_sum < class1_sum:
                classes[counter] = 1
            counter += 1
        return classes

    def custom_naive_bayes_score(data_set, log_features_class0, log_features_class1, class0_log, class1_log):

        classed_0 = classify(log_features_class0, class0_log, log_features_class1, class1_log, np.where(data_set[:, FEATS] == 0))
        classed_1 = classify(log_features_class0, class0_log, log_features_class1, class1_log, np.where(data_set[:, FEATS] == 1))

        predicted = np.append(classed_0, classed_1)
        hit_mask = np.append(np.zeros(classed_0.shape), np.ones(classed_1.shape))
        error_mask = np.append(np.ones(classed_0.shape), np.zeros(classed_1.shape))
        hits = get_score(predicted, hit_mask)
        hit_percentage = get_score(predicted, hit_mask, True) * 100
        errors = get_score(predicted, error_mask)
        error_percentage = get_score(predicted, error_mask, True)

        return error_percentage

    class0_train_points, class1_train_points = separate_classes(_train_data[:, 0:FEATS], _train_data[:, FEATS])

    train_log_features_class0 = create_distrib_matrix(class0_train_points, _train_data[:, 0:FEATS])

    train_log_features_class1 = create_distrib_matrix(class1_train_points, _train_data[:, 0:FEATS])

    test_log_features_class0 = create_distrib_matrix(class0_train_points, _test_data[:, 0:FEATS])

    test_log_features_class1 = create_distrib_matrix(class1_train_points, _test_data[:, 0:FEATS])

    total = _train_data.shape[0]
    log_class0 = np.log(float(class0_train_points.shape[0]) / total)
    log_class1 = np.log(float(class1_train_points.shape[0]) / total)
    return\
        custom_naive_bayes_score(_train_data, train_log_features_class0, train_log_features_class1, log_class0, log_class1),\
        custom_naive_bayes_score(_test_data, test_log_features_class0, test_log_features_class1, log_class0, log_class1)


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
            fold_train_err, fold_va_err = clf_function(_train_data[tr_ix], _train_data[va_ix], p)
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
    return clf.score(_test_data[:, 0:4], _test_data[:, 4])


def approximate_normal_test():
    pass


def mc_nemars_test():
    pass


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


print("C found: " + str(c))
print("H found: " + str(h))
gaussian_naive_bayes(train_data, test_data)
