import tp2_aux as utils

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from pandas.plotting import radviz, scatter_matrix
import seaborn as sns

import numpy as np
import pickle
import pandas as pd
from functools import partial
from tqdm import tqdm
from os import path

"""def plot(x, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['#ffffff', '#8b0000', '#00a5ff', '#50c878']
    values = np.unique(y)
    for class_value in values:
        x1_plot = x[np.where(y == class_value)][:, 0]
        x2_plot = x[np.where(y == class_value)][:, 1]
        ax.scatter(x1_plot, x2_plot, c=colors[int(class_value)], s=10)
    # ax.scatter(x, y, '.r')
    # ax.scatter(y, x, '.b')
    # ax.set_title(title)
    # ax.set_xlabel(x_title)
    ax.legend(['UNKNOWN', 'CLASS 1', 'CLASS 2', 'CLASS 3'])
    plt.savefig("myplot.png")
    plt.show()
    plt.close(fig)
"""


def plot_heatmap(data):
    plt.figure(figsize=(20, 15))
    corr = data.iloc[:, :-1].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr, annot=True, mask=mask, square=True, cmap=plt.get_cmap("RdYlGn_r"))
    plt.savefig('heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()


def standardize(_data):
    return (_data - np.mean(_data)) / np.std(_data)


def normalize(_data):
    _min = np.min(_data)
    return (_data - _min) / (np.max(_data) - _min)


def f_test(_feats, _labels):
    return f_classif(_feats, _labels)[0]


def selectKBest(_data, _filter, _k=18):
    return _data[:, np.argsort(_filter)[:_k][::-1]]


def selectLowestCorr(correlation_matrix, max_correlation=0.5):
    low_corr_feats = []
    high_corr_feats = []
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if j == 0:
                low_corr_feats.append(i)
            elif i >= j:
                break
            elif i in high_corr_feats:
                continue
            else:
                c_value = correlation_matrix.iloc[i, j]
                if c_value < max_correlation:
                    low_corr_feats.append(j)
                else:
                    high_corr_feats.append(j)
    print(low_corr_feats)


np.set_printoptions(precision=4, suppress=True)
tqdm = partial(tqdm, position=0, leave=True)

labels = np.loadtxt("labels.txt", delimiter=",")
LABELED = np.where(labels[:, -1] != 0)

if path.exists("feats.p"):
    f = open("feats.p", "rb")
    feats = pickle.load(f)
else:
    pca = PCA(n_components=6)
    tsne = TSNE(n_components=6, method='exact')
    iso = Isomap(n_components=6)
    print("Extracting features...")
    img_mat = utils.images_as_matrix()
    feats = np.column_stack([pca.fit_transform(img_mat), tsne.fit_transform(img_mat), iso.fit_transform(img_mat)])
    pickle.dump(feats, open("feats.p", "wb"))
print("Features extracted")

plot_data = feats
columns = [f'{num}' for num in range(plot_data.shape[1])]
columns.append("class")
dataframe = pd.DataFrame(np.column_stack([plot_data, labels[:, -1]]), columns=columns)
corr = dataframe.iloc[:, :-1].corr()
selectLowestCorr(corr)
# selectLowestCorr(dataframe.iloc[:, :-1].corr())
plot_heatmap(dataframe)

# Create clusters
# kmeans_clusters = KMeans()
# dbscan_clusters = DBSCAN()

# Plot clusters to features


# Evaluate clusters


# Plot evaluation metrics
