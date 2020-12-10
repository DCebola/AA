import tp2_aux as utils
import numpy as np
import pandas as pd
import pickle
import os.path as path
from math import sqrt
from functools import reduce

# Sklearn
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans, DBSCAN, FeatureAgglomeration
from sklearn.neighbors import KNeighborsClassifier

# SciPy
from scipy.signal import find_peaks_cwt

# Skimage
from skimage.io import imsave, imread
from skimage.transform import resize

# Plots
import matplotlib.pyplot as plt
import seaborn as sns


def plot_joint_plot(_data, name, title):
    plt.figure(figsize=(20, 15))
    plt.title(title)
    sns.jointplot(data=_data, x=_data.columns[0], y=_data.columns[1], hue="class", kind="kde", palette=PALETTE[:_data["class"].unique().size])
    plt.savefig('plots/' + name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_paired_density_and_scatter_plot(_data, name, title):
    plt.figure(figsize=(20, 15))
    plt.title(title)
    g = sns.PairGrid(_data, hue="class", palette=PALETTE[:_data["class"].unique().size])
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    plt.savefig('plots/' + name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_heatmap(_data, name, title):
    plt.figure(figsize=(20, 15))
    plt.title(title)
    _corr = _data.corr()
    mask = np.zeros_like(_corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(_corr, vmax=1, vmin=-1, annot=True, mask=mask, square=True, cmap=plt.get_cmap("RdYlGn_r"))

    plt.savefig('plots/' + name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_5dist(_data, _valleys, name, title):
    plt.figure(figsize=(20, 15))
    plt.title(title)
    _x = np.linspace(0, len(_data), len(_data))
    plt.plot(_x, _data, '-b')
    plt.xlabel("Point")
    plt.ylabel("Distance")
    plt.scatter(_valleys, _data[_valleys], marker="x", color=['r'], alpha=1, s=50)
    plt.savefig('plots/' + name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_metrics(_data, name, parameter, title):
    plt.figure(figsize=(20, 15))
    plt.title(title)
    sns.lineplot(data=_data, x="x", y="y", hue="metric", palette=PALETTE[:_data["metric"].unique().size])
    sns.scatterplot(data=_data, x="x", y="y", hue="metric", legend=False, palette=PALETTE[:_data["metric"].unique().size])
    plt.xlabel(parameter)
    plt.ylabel("Score")
    plt.savefig('plots/' + name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def standardize(_data):
    return (_data - np.mean(_data)) / np.std(_data)


def normalize(_data):
    _min = np.min(_data)
    return (_data - _min) / (np.max(_data) - _min)


def f_test(_feats, _labels):
    return f_classif(_feats, _labels)[0]


def filterKBest(_data, _filter, _k=18):
    return np.argsort(_filter)[:_k][::-1]


def filter_low_high_corr(correlation_matrix, max_correlation=0.5):
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


def calculate_confusion_matrix(_feats, _predicted_labels, _true_labels):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(_feats.shape[0]):
        for j in range(i + 1, _feats.shape[0]):
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
    return tp, tn, fn, fp


def cluster_eval(_feats, _predicted_labels, _true_labels, _x):
    labeled_feats = _feats.iloc[LABELED]
    tp, tn, fn, fp = calculate_confusion_matrix(labeled_feats, _predicted_labels, _true_labels)
    n_pairs = (labeled_feats.shape[0] * (labeled_feats.shape[0] - 1)) / 2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    rand_index = (tp + tn) / n_pairs
    f1 = 2 * (precision * recall) / (precision + recall)
    try:
        _silhouette_score = silhouette_score(_feats.to_numpy(), _predicted_labels)
    except ValueError:
        _silhouette_score = -1
    _adjusted_rand_score = adjusted_rand_score(_true_labels, _predicted_labels[LABELED])
    return [
        ('silhouette_score', _x, _silhouette_score),
        ('precision_score', _x, precision),
        ('recall_score', _x, recall),
        ('f1_score', _x, f1),
        ('adjusted_rand_score', _x, _adjusted_rand_score),
        ('rand_index', _x, rand_index)
    ]


def generate_KMeans_clusters(_data, n_clusters, _true_labels):
    _results = []
    for n in range(2, n_clusters + 1):
        _clusters = KMeans(n_clusters=n).fit_predict(_data)
        _plt_data = _data.iloc[:, filterKBest(_data, f_test(_data, _clusters), 18)]
        plot_paired_density_and_scatter_plot(pd.concat([_plt_data.iloc[:, :6], pd.Series(_clusters, name='class')], axis=1),
                                             "kmeans/clusters_paired_densities_" + str(n), "Kmeans w/ " + str(n) + " clusters")
        for m in cluster_eval(_data, _clusters, _true_labels, n):
            _results.append(m)
    return _results


def generate_DBSCAN_clusters(_data, _valleys_dists, _true_labels):
    _results = []
    i = 0
    for v in _valleys_dists:
        i += 1
        _clusters = DBSCAN(eps=v).fit_predict(_data)
        _plt_data = _data.iloc[:, filterKBest(_data, f_test(_data, _clusters), 18)]
        plot_paired_density_and_scatter_plot(pd.concat([_plt_data.iloc[:, :6], pd.Series(_clusters, name='class')], axis=1),
                                             "dbscan/clusters_paired_densities_" + str(i), "DBSCAN w/ ε = " + str(v))
        for m in cluster_eval(_data, _clusters, _true_labels, v):
            _results.append(m)
    return _results


def get_factors(value):
    factors = []
    for i in range(1, int(value ** 0.5) + 1):
        if value % i == 0:
            factors.append((int(value / i), i))
    factors.reverse()
    return factors[0]


def save_images(_path, _images):
    i = 0
    for _image in _images:
        imsave(_path + str(i) + ".png", resize(np.reshape(_image, get_factors(_image.shape[0])), (50, 50),
                                               order=0, preserve_range=True, anti_aliasing=False).astype('uint8'))
        i += 1


def agglomerate_images_pixels(_images):
    print(_images[0].shape)
    _agglomerated_feats = FeatureAgglomeration(connectivity=grid_to_graph(50, 50), n_clusters=2)
    _agglomerated_feats.fit(_images)
    _reduced = _agglomerated_feats.transform(_images)
    _restored = _agglomerated_feats.inverse_transform(_reduced)
    save_images("reduced_images/", _reduced)
    save_images("restored_images/", _restored)


np.set_printoptions(precision=4, suppress=True)
sns.set_style("ticks")
PALETTE = sns.color_palette("muted", 12)
NUM_ITER = 5
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
    agglomerate_images_pixels(img_mat)
    feats = np.column_stack([pca.fit_transform(img_mat), tsne.fit_transform(img_mat), iso.fit_transform(img_mat)])
    # pickle.dump(feats, open("feats.p", "wb"))
"""
DATA_COLS = [f'f_{num}' for num in range(feats.shape[1])]
DATA_COLS.append("class")
METRICS_COLS = ["metric", "x", "y"]
data_df = pd.DataFrame(normalize(np.column_stack([feats, labels[:, -1]])), columns=DATA_COLS)
print("Features extracted")

# -----------------------------------------------------------------Feature selection------------------------------------------------------------------
# Finding low and highly correlated features
LOW_CORR, HIGH_CORR = filter_low_high_corr(data_df.iloc[:, :-1].corr())
plot_heatmap(data_df.iloc[:, :-1], "full_heatmap", "Features Heatmap")
plot_heatmap(data_df.iloc[:, LOW_CORR], "low_corr_heatmap", "Low Correlated Features Heatmap")
# plot_data = selectKBest(feats, f_test(feats[LABELED], labels[LABELED][:,-1]), 18)

# Sequential backward elimination

# KMeans
# DBSCAN
# Bisecting K-Means

# -----------------------------------------------------------------Cluster Generation-----------------------------------------------------------------

# KMeans
kmeans_cluster_results_df = pd.DataFrame(generate_KMeans_clusters(data_df.iloc[:, :-1], NUM_ITER + 1, labels[LABELED][:, 1]), columns=METRICS_COLS)
plot_metrics(kmeans_cluster_results_df, "kmeans/kmeans_cluster_metrics", "Clusters", "KMeans Cluster Metrics")

# DBSCAN

# kn = KNeighborsClassifier()
# kn.fit(plot_data, np.zeros(plot_data.shape[0]))
# kneighbors = np.sort(np.array(kn.kneighbors()[0]).flatten())[::-1]

plot_data = feats[:, HIGH_CORR]
_5dists = np.sort(np.linalg.norm(plot_data - plot_data[:, None], axis=-1), axis=-1)[::-1][:, 4]
_5dists = np.sort(_5dists)[::-1]

valleys = find_peaks_cwt(_5dists * (-1), np.arange(1, 4))
valleys_dists = _5dists[valleys]
plot_5dist(_5dists, valleys[0], "dbscan/5-dists", "5 Distances")

dbscan_cluster_results_df = pd.DataFrame(generate_DBSCAN_clusters(data_df.iloc[:, :-1], valleys_dists[:NUM_ITER], labels[LABELED][:, 1]),
                                         columns=METRICS_COLS)
plot_metrics(dbscan_cluster_results_df, "dbscan/dbscan_cluster_metrics", "ε", "DBSCAN Cluster Metrics")

# Bisecting K-Means
"""
