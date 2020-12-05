import tp2_aux as utils

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from pandas.plotting import radviz,scatter_matrix
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


def plot_(data):
    plt.figure(figsize=(20, 15))
    corr = data.iloc[:, :-1].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr, annot=True, mask=mask, square=True, cmap=plt.get_cmap("RdYlGn_r"))
    plt.savefig('.png', dpi=200, bbox_inches='tight')
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

feats_by_relevance = f_test(feats[LABELED], labels[LABELED][:, -1])
#plot_data = selectKBest(feats, feats_by_relevance, 18)
plot_data = feats
columns = [f'feat_{num}' for num in range(plot_data.shape[1])]
columns.append("class")
dataframe = pd.DataFrame(np.column_stack([plot_data, labels[:, -1]]), columns=columns)
plot_(dataframe)
# plot(data, labels[:, -1])

# Create clusters
# kmeans_clusters = KMeans()
# dbscan_clusters = DBSCAN()

# Plot clusters to features


# Evaluate clusters


# Plot evaluation metrics
