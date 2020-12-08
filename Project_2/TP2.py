import tp2_aux as utils
import numpy as np
import pandas as pd
import pickle
import os.path as path

# Sklearn
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier

# SciPy
from scipy.signal import find_peaks_cwt

# Plots
import matplotlib.pyplot as plt
import seaborn as sns


def plot_joint_plot(data, name):
    plt.title(name)
    sns.jointplot(data, hue="class")
    sns.jointplot(data, hue="class", kind="kde")
    plt.savefig(name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_paired_density_and_scatter_plot(data, name):
    plt.figure(figsize=(20, 15))
    plt.title(name)
    g = sns.pairplot(data, hue="class", markers=["o", "s", "D"])
    g.map_lower(sns.kdeplot, hue="class")
    plt.savefig(name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_heatmap(data, name):
    plt.figure(figsize=(20, 15))
    plt.title(name)
    _corr = data.corr()
    mask = np.zeros_like(_corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(_corr, vmax=1, vmin=-1, annot=True, mask=mask, square=True, cmap=plt.get_cmap("RdYlGn_r"))

    plt.savefig(name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_5dist(_data, _valleys, name):
    plt.title(name)
    _x = np.linspace(0, len(_data), len(_data))
    plt.plot(_x, _data, '-b')
    plt.scatter(_valleys, _data[_valleys], marker="x", color=['r'], alpha=1, s=50)
    plt.savefig(name)
    plt.close()


def plot_metrics(data, name, parameter):
    plt.figure(figsize=(20, 15))
    plt.title(name)
    g = sns.FacetGrid(data, col="metric", hue='metric', col_wrap=3, margin_titles=True)
    g.map_dataframe(sns.lineplot, x='x', y='y')
    g.set_axis_labels("Metric Value", parameter)
    g.set_titles(col_template="{col_name}")
    plt.savefig(name + '.png', dpi=200, bbox_inches='tight')
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


def cluster_eval(_feats, _predicted_labels, _true_labels, _y):
    # adjusted_rand_score, precision_score, recall_score, f1_score, silhouette_score
    labeled_feats = _feats[LABELED]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(labeled_feats.shape[0]):
        for j in range(i + 1, labeled_feats.shape[0]):
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
    n_pairs = (labeled_feats.shape[0] * (labeled_feats.shape[0] - 1)) / 2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    rand_index = (tp + tn) / n_pairs
    f1 = 2 * (precision * recall) / (precision + recall)
    _silhouette_score = silhouette_score(_feats, _predicted_labels)
    _adjusted_rand_score = adjusted_rand_score(_true_labels, _predicted_labels[LABELED])
    print({"silhouette_score": _silhouette_score,
           "precision_score": precision,
           "recall_score": recall,
           "f1_score": f1,
           "adjusted_rand_score": _adjusted_rand_score,
           "rand_index": rand_index})
    # after (...).append( (...) , ignore_index=True)
    return [
        pd.Series(['silhouette_score', _silhouette_score, _y], index=METRICS_COLS),
        pd.Series(['precision_score', precision, _y], index=METRICS_COLS),
        pd.Series(['recall_score', recall, _y], index=METRICS_COLS),
        pd.Series(['f1_score', f1, _y], index=METRICS_COLS),
        pd.Series(['adjusted_rand_score', _adjusted_rand_score, _y], index=METRICS_COLS),
        pd.Series(['rand_index', rand_index, _y], index=METRICS_COLS)
    ]


def generate_KMeans_clusters(_data, n_clusters, _true_labels):
    _results = []
    for n in range(n_clusters + 1):
        _results.append(cluster_eval(_data, KMeans(n_clusters=n).fit_predict(_data), _true_labels, n))
    return _results


def generate_DBSCAN_clusters(_data, _valleys_dists, _true_labels):
    _results = []
    for v in _valleys_dists:
        _results.append(cluster_eval(_data, DBSCAN(eps=v).fit_predict(_data), _true_labels, v))
    return _results


np.set_printoptions(precision=4, suppress=True)
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

DATA_COLS = [f'{num}' for num in range(feats.shape[1])].append("class")
METRICS_COLS = ["metric", "x", "y"]
data_df = pd.DataFrame(np.column_stack([feats, labels[:, -1]]), columns=DATA_COLS)
print("Features extracted")

# -----------------------------------------------------------------Feature selection------------------------------------------------------------------
# Finding low and highly correlated features
LOW_CORR, HIGH_CORR = filter_low_high_corr(data_df.iloc[:, :-1].corr())
plot_heatmap(data_df.iloc[:, :-1], "full_heatmap")
plot_heatmap(data_df.iloc[:, LOW_CORR], "low_corr_heatmap")
# plot_data = selectKBest(feats, f_test(feats[LABELED], labels[LABELED][:,-1]), 18)

# Sequential backward elimination

# KMeans
# DBSCAN
# Bisecting K-Means

# -----------------------------------------------------------------Cluster Generation-----------------------------------------------------------------

# KMeans
# DBSCAN

# kn = KNeighborsClassifier()
# kn.fit(plot_data, np.zeros(plot_data.shape[0]))
# kneighbors = np.sort(np.array(kn.kneighbors()[0]).flatten())[::-1]

plot_data = feats[:, HIGH_CORR]
_5dists = np.sort(np.linalg.norm(plot_data - plot_data[:, None], axis=-1), axis=-1)[::-1][:, 4]
_5dists = np.sort(_5dists)[::-1]

valleys = find_peaks_cwt(_5dists * (-1), np.arange(1, 4))
valleys_dists = _5dists[valleys]
plot_5dist(_5dists, valleys[0], "5-dists")

# Bisecting K-Means
