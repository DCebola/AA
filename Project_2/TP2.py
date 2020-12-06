import tp2_aux as utils

from sklearn.metrics import adjusted_rand_score, precision_score, recall_score, f1_score, silhouette_score
from scipy.signal import find_peaks_cwt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pandas.plotting import radviz, scatter_matrix, parallel_coordinates
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


def plot_heatmap(data, name):
    plt.figure(figsize=(20, 15))
    _corr = data.corr()
    mask = np.zeros_like(_corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(_corr, vmax=1, vmin=-1, annot=True, mask=mask, square=True, cmap=plt.get_cmap("RdYlGn_r"))

    plt.savefig(name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_parallel_coordinates(_data):
    parallel_coordinates(_data, 'class')
    plt.savefig('parallel.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_5dist(_data, _valleys, name):
    _x = np.linspace(0, len(_data), len(_data))
    plt.plot(_x, _data, '-b')
    plt.scatter(_valleys, _data[_valleys], marker="x", color=['r'], alpha=1, s=50)
    plt.savefig(name)
    plt.close()
    pass


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
    to_keep = [i for i in range(correlation_matrix.shape[1])]
    to_remove = []
    for i in range(correlation_matrix.shape[1]):
        if i in to_remove:
            continue
        else:
            for j in range(i + 1, correlation_matrix.shape[0]):
                if j in to_remove:
                    continue
                else:
                    c_value = correlation_matrix.iloc[i, j]
                    if c_value > max_correlation:
                        if j in to_keep:
                            to_keep.remove(j)
                            to_remove.append(j)
    return to_keep, to_remove


def cluster_eval(_feats, _predicted_labels, _true_labels):
    #adjusted_rand_score, precision_score, recall_score, f1_score, silhouette_score
    labeled_feats = _feats[LABELED]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(labeled_feats.shape[0]):
        for j in range(i+1, labeled_feats.shape[0]):
            if i == j:
                continue
            else:
                if _predicted_labels[LABELED][i] == _predicted_labels[LABELED][j]:
                    if _true_labels[i] == _true_labels[j]:
                        tp += 1
                    else:
                        fp += 1
                elif _true_labels[i] == _true_labels[j]:
                    fn += 1
                else:
                    tn += 1
    n_pairs = (labeled_feats.shape[0] * (labeled_feats.shape[0]-1))/2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    rand_index = (tp + tn) / n_pairs
    f1 = 2 * (precision*recall)/(precision+recall)
    scores = {"silhouette_score": silhouette_score(_feats, _predicted_labels)
            , "precision_score": precision
            , "recall_score": recall
            , "f1_score": f1
            , "adjusted_rand_score": adjusted_rand_score(_true_labels, _predicted_labels[LABELED])
            , "rand_index": rand_index}
    return scores

def generate_KMeans_clusters(_data, n_clusters):
    for n in range(n_clusters + 1):
        cluster_eval(KMeans(n_clusters=n).fit_predict(_data))


def generate_DBSCAN_clusters(_data, _valleys_dists):
    for v in _valleys_dists:
        cluster_eval(DBSCAN(eps=v).fit_predict(_data))


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

#plot_data = selectKBest(feats, f_test(feats[LABELED], labels[LABELED][:,-1]), 18)
plot_data = feats

columns = [f'{num}' for num in range(plot_data.shape[1])]
columns.append("class")
dataframe = pd.DataFrame(np.column_stack([plot_data, labels[:, -1]]), columns=columns)
low_corr, high_corr = selectLowestCorr(dataframe.iloc[:, :-1].corr())
plot_heatmap(dataframe.iloc[:, :-1], "full_heatmap")
plot_heatmap(dataframe.iloc[:, low_corr], "low_corr_heatmap")

# kn = KNeighborsClassifier()
# kn.fit(plot_data, np.zeros(plot_data.shape[0]))
# kneighbors = np.sort(np.array(kn.kneighbors()[0]).flatten())[::-1]

plot_data = plot_data[:,high_corr]
_5dists = np.sort(np.linalg.norm(plot_data - plot_data[:, None], axis=-1), axis=-1)[::-1][:, 4]
_5dists = np.sort(_5dists)[::-1]

valleys = find_peaks_cwt(_5dists * (-1), np.arange(1, 4))
valleys_dists = _5dists[valleys]
plot_5dist(_5dists, valleys[0], "5-dists")

kmeans = KMeans(n_clusters=3)
final_labels = kmeans.fit_predict(plot_data)

print(cluster_eval(plot_data, final_labels, labels[LABELED][:, -1]))

# Plot clusters to features


# Evaluate clusters


# Plot evaluation metrics
