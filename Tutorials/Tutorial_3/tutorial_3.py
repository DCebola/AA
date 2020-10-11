import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import t3_aux
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import math


def calc_fold(feats, X, Y, train_ix, valid_ix, C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix, :feats], Y[train_ix])
    prob = reg.predict_proba(X[:, :feats])[:, 1]
    squares = (prob - Y) ** 2
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])


data = np.loadtxt('data.txt', delimiter=',')
data = shuffle(data)
Ys = data[:, 0]
Xs = data[:, 1:]
means = np.mean(Xs, axis=0)
stdevs = np.std(Xs, axis=0)
Xs = (Xs - means) / stdevs
Xs = t3_aux.poly_16features(Xs)
X_r, X_t, Y_r, Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify=Ys)

'''
folds = 10
best_feat = 0
best_val_err = 1000000
kf = StratifiedKFold(n_splits=folds)
errors = np.zeros((13,2))
for feats in range(2,16):
    tr_err = va_err = 0
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        r,v = calc_fold(feats,X_r,Y_r,tr_ix,va_ix)
        tr_err += r
        va_err += v
    if(va_err/folds<best_val_err  ):
        best_val_err = va_err/folds
        best_feat = feats
    print(feats,':', tr_err/folds,va_err/folds)
    errors[feats-2,:] = errors[feats-2,:] + (tr_err/folds,va_err/folds)
    

t3_aux.create_plot(X_r, Y_r, X_t, Y_t, best_feat,1e12)
'''

folds = 10
best_feat = 0
best_val_err = 1000000
kf = StratifiedKFold(n_splits=folds)
errors = np.zeros((13, 2))
tr_err = va_err = 0
c = 1
logCToTrainErr = np.zeros((20, 2))
logCToValErr = np.zeros((20, 2))

best_c = 0
for i in range(1, 21):
    tr_err = va_err = 0
    for tr_ix, va_ix in kf.split(Y_r, Y_r):
        r, v = calc_fold(16, X_r, Y_r, tr_ix, va_ix, c)
        tr_err += r
        va_err += v
    logCToTrainErr[i - 1, :] = logCToTrainErr[i - 1, :] + (math.log(c), tr_err / folds)
    logCToValErr[i - 1, :] = logCToValErr[i - 1, :] + (math.log(c), va_err / folds)

    if va_err / folds < best_val_err:
        best_val_err = va_err / folds
        best_c = c
    c = c * 2

pxs = np.linspace(min(logCToValErr[:, 0]), max(logCToValErr[:, 0]), 100)
max_y = max([max(logCToValErr[:, 1]), max(logCToTrainErr[:, 1])])
plt.axis([1, 16, 0, max_y])

plt.plot(logCToTrainErr[:, 0], logCToTrainErr[:, 1], '-b')
plt.plot(logCToValErr[:, 0], logCToValErr[:, 1], '-r')

plt.show()
# t3_aux.create_plot(X_r, Y_r, X_t, Y_t,16,best_c)
