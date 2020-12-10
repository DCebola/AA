import tp2_aux as utils
import numpy as np
import numpy.matlib as npm
import pandas as pd
import pickle
from os import path, mkdir
from math import sqrt
from functools import reduce
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans, DBSCAN, FeatureAgglomeration
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import find_peaks_cwt
from skimage.io import imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = sns.color_palette("muted", 12)
np.set_printoptions(precision=4, suppress=True)
sns.set_style("ticks")


def create_dir(_path):
    if not path.exists(_path):
        mkdir(_path)


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
    _cluster_results = []
    for n in range(2, n_clusters + 1):
        _clusters = KMeans(n_clusters=n).fit_predict(_data)
        _cluster_results.append((_clusters, _data.iloc[:, filterKBest(_data, f_test(_data, _clusters), 18)], pd.Series(_clusters, name='class')))
        for m in cluster_eval(_data, _clusters, _true_labels, n):
            _results.append(m)
    return _results, _cluster_results


def generate_DBSCAN_clusters(_data, _valleys_dists, _true_labels):
    _results = []
    _cluster_results = []
    i = 0
    for v in _valleys_dists:
        i += 1
        _clusters = DBSCAN(eps=v).fit_predict(_data)
        _cluster_results.append((_clusters, _data.iloc[:, filterKBest(_data, f_test(_data, _clusters), 18)], pd.Series(_clusters, name='class'), v))
        for m in cluster_eval(_data, _clusters, _true_labels, v):
            _results.append(m)
    return _results, _cluster_results


def findElbow(curve):
    nPoints = curve.shape[0]
    allCoord = np.vstack((range(curve.shape[0]), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * npm.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    return allCoord[np.argmax(distToLine)]


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


def agglomerate_images_pixels(_images, N=18):
    _agglomerated_feats = FeatureAgglomeration(connectivity=grid_to_graph(50, 50), n_clusters=N)
    _agglomerated_feats.fit(_images)
    _reduced = _agglomerated_feats.transform(_images)
    _restored = _agglomerated_feats.inverse_transform(_reduced)
    create_dir("reduced_images/" + str(N))
    create_dir("restored_images/" + str(N))
    save_images("reduced_images/" + str(N) + "/", _reduced)
    save_images("restored_images/" + str(N) + "/", _restored)
    return _reduced, _restored


def extract_features(_images):
    return np.column_stack([PCA(n_components=6).fit_transform(_images),
                            TSNE(n_components=6, method='exact').fit_transform(_images),
                            Isomap(n_components=6).fit_transform(_images)])


def get_original_feats_data(_images):
    if path.exists("original_feats.p"):
        print("Loading features...")
        _original_feats = pickle.load(open("original_feats.p", "rb"))
        print("Loaded original features...")
    else:
        print("Extracting features...")
        _original_feats = extract_features(_images)
        pickle.dump(_original_feats, open("original_feats.p", "wb"))
        print("Extracted original features...")
    return _original_feats


def get_reduced_feats_data(_images, N=18):
    if path.exists("reduced_feats.p"):
        print("Loading features...")
        _reduced_feats = pickle.load(open("reduced_feats.p", "rb"))
        print("Loaded reduced features...")
        _restored_feats = pickle.load(open("restored_feats.p", "rb"))
        print("Loaded restored features...")
    else:
        print("Reducing images...")
        _reduced_images, _restored_images = agglomerate_images_pixels(_images, N)
        _reduced_feats = extract_features(_reduced_images)
        pickle.dump(_reduced_feats, open("reduced_feats.p", "wb"))
        print("Extracted reduced features...")
        _restored_feats = extract_features(_restored_images)
        pickle.dump(_restored_feats, open("restored_feats.p", "wb"))
        print("Extracted restored features...")
    return _reduced_feats, _restored_feats


def experiment(_path, _feats, _labels, cluster_iter):
    print(150 * "_")
    create_dir("plots/" + _path)
    create_dir("plots/" + _path + "/kmeans")
    create_dir("plots/" + _path + "/kmeans/clusters")
    create_dir("plots/" + _path + "/dbscan")
    create_dir("plots/" + _path + "/dbscan/clusters")
    DATA_COLS = [f'f_{num}' for num in range(_feats.shape[1])]
    DATA_COLS.append("class")
    METRICS_COLS = ["metric", "x", "y"]
    data_df = pd.DataFrame(normalize(np.column_stack([_feats, _labels[:, -1]])), columns=DATA_COLS)

    # -----------------------------------------------------------------Feature selection--------------------------------------------------------------
    # Finding low and highly correlated features
    LOW_CORR, HIGH_CORR = filter_low_high_corr(data_df.iloc[:, :-1].corr())
    print("Found correlation between features.")

    # Sequential backward elimination
    # KMeans
    # DBSCAN
    # Bisecting K-Means

    # -----------------------------------------------------------------Cluster Generation-------------------------------------------------------------

    # KMeans
    kmeans_metrics, kmeans_clusters = generate_KMeans_clusters(data_df.iloc[:, :-1], cluster_iter + 1, labels[LABELED][:, 1])
    kmeans_metrics_df = pd.DataFrame(kmeans_metrics, columns=METRICS_COLS)
    print("Generated kmeans clusters.")

    # DBSCAN
    kn = KNeighborsClassifier()
    kn.fit(_feats, np.zeros(_feats.shape[0]))
    _5dists = np.sort(np.array(kn.kneighbors()[0]).flatten())[::-1]
    # _5dists = np.sort(np.linalg.norm(_feats - _feats[:, None], axis=-1), axis=-1)[::-1][:, 4]
    # _5dists = np.sort(_5dists)[::-1]
    # valleys = find_peaks_cwt(_5dists * (-1), np.arange(1, 5))
    # valleys_dists = _5dists[valleys]
    best_eps = findElbow(_5dists)
    eps_range = []
    for num in range(cluster_iter):
        eps_range.append(best_eps[1] - num * 250)
    dbscan_metrics, dbscan_clusters = generate_DBSCAN_clusters(data_df.iloc[:, :-1], eps_range, labels[LABELED][:, 1])
    dbscan_metrics_df = pd.DataFrame(dbscan_metrics, columns=METRICS_COLS)
    print("Generated dbscan clusters.")

    # Bisecting K-Means

    # -----------------------------------------------------------------Plot Results-------------------------------------------------------------------

    # Features correlations
    plot_heatmap(data_df.iloc[:, :-1], _path + "/full_heatmap", "Features Heatmap")
    plot_heatmap(data_df.iloc[:, LOW_CORR], _path + "/low_corr_heatmap", "Low Correlated Features Heatmap")
    print("Plotted features' heatmaps.")

    # Kmeans results
    plot_metrics(kmeans_metrics_df, _path + "/kmeans/kmeans_cluster_metrics", "Clusters", "KMeans Cluster Metrics")
    i = 0
    for _run_data in kmeans_clusters:
        utils.report_clusters(_labels[:, 0], _run_data[0], "plots/" + _path + "/kmeans/clusters/clusters_" + str(i + 1) + ".html")
        plot_paired_density_and_scatter_plot(pd.concat([_run_data[1].iloc[:, :6], _run_data[2]], axis=1),
                                             _path + "/kmeans/clusters_paired_densities_" + str(i), "Kmeans w/ " + str(i) + " clusters")
        i += 1
        print("Plotted kmeans cluster - " + str(i))

    # DBSCAN results
    plot_5dist(_5dists, int(best_eps[0]), _path + "/dbscan/5-dists", "5 Distances")
    plot_metrics(dbscan_metrics_df, _path + "/dbscan/dbscan_cluster_metrics", "ε", "DBSCAN Cluster Metrics")
    i = 0
    for _run_data in dbscan_clusters:
        utils.report_clusters(_labels[:, 0], _run_data[0], "plots/" + _path + "/dbscan/clusters/clusters_" + str(i + 1) + ".html")
        plot_paired_density_and_scatter_plot(pd.concat([_run_data[1].iloc[:, :6], _run_data[2]], axis=1),
                                             _path + "/dbscan/clusters_paired_densities_" + str(i), "DBSCAN w/ ε = " + str(_run_data[3]))
        i += 1
        print("Plotted dbscan cluster - " + str(i))


labels = np.loadtxt("labels.txt", delimiter=",")
LABELED = np.where(labels[:, -1] != 0)
images = utils.images_as_matrix()
CLUSTER_ITERATIONS = 5
create_dir("plots")
create_dir("reduced_images")
create_dir("restored_images")

original_feats = get_original_feats_data(images)
experiment("original", original_feats, labels, CLUSTER_ITERATIONS)

# reduced_feats, restored_feats = get_reduced_feats_data(images)
# experiment("reduced", reduced_feats, labels, CLUSTER_ITERATIONS)
# experiment("restored", restored_feats, labels, CLUSTER_ITERATIONS)
