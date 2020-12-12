import tp2_aux as utils
import numpy as np
import pandas as pd
import pickle
from os import path, mkdir
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans, DBSCAN, FeatureAgglomeration, AgglomerativeClustering
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
    sns.jointplot(data=_data, x=_data.columns[0], y=_data.columns[1], hue="class", kind="kde",
                  palette=PALETTE[:_data["class"].unique().size])
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


def plot_cluster_bar_metrics(_data, name, title):
    plt.figure(figsize=(20, 15))
    plt.title(title)
    g = sns.barplot(data=_data, x="metric", y="y", palette=PALETTE[:_data["metric"].unique().size])
    g.set_xticklabels(g.get_xticklabels(), rotation=30)
    i = 0
    for _y in _data.get('y'):
        g.text(i, round(_y, 4), round(_y, 4), color='black', ha="center", fontsize='x-large')
        i += 1
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig("plots/" + name + '.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_cluster_line_metrics(_data, name, parameter, title):
    plt.subplots(figsize=(20, 15))
    plt.title(title)
    sns.lineplot(data=_data, x="x", y="y", hue="metric", palette=PALETTE[:_data["metric"].unique().size])
    sns.scatterplot(data=_data, x="x", y="y", hue="metric", legend=False,
                    palette=PALETTE[:_data["metric"].unique().size])
    plt.xlabel(parameter)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig("plots/" + name + '.png', dpi=200, bbox_inches='tight')
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
                else:
                    if _true_labels[i] == _true_labels[j]:
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
        _silhouette_score = 0
    _adjusted_rand_score = adjusted_rand_score(_true_labels, _predicted_labels[LABELED])
    return [
        ('Silhouette Score', _x, _silhouette_score),
        ('Precision Score', _x, precision),
        ('Recall', _x, recall),
        ('F1-Measure', _x, f1),
        ('Adjusted Random Score', _x, _adjusted_rand_score),
        ('Random Index', _x, rand_index)
    ]


def generate_KMeans_clusters(_data, n_clusters, _true_labels):
    _metrics = []
    _clusters = []
    for n in range(2, n_clusters + 1):
        _clusters = KMeans(n_clusters=n).fit_predict(_data)
        for m in cluster_eval(_data, _clusters, _true_labels, n):
            _metrics.append(m)
    return _metrics, _clusters


def generate_Agglomerative_clusters(_data, n_clusters, _true_labels):
    _metrics = []
    _clusters = []
    for n in range(2, n_clusters + 1):
        _clusters = AgglomerativeClustering(n_clusters=n).fit_predict(_data)
        for m in cluster_eval(_data, _clusters, _true_labels, n):
            _metrics.append(m)
    return _metrics, _clusters


def generate_Bisecting_KMeans_clusters(_data, max_iter, _true_labels):
    _metrics = []
    _clusters = []
    for n in range(1, max_iter + 1):
        _clusters = MyBisectingKMeans(n_iter=n).fit_predict(_data)
        for m in cluster_eval(_data, _clusters, _true_labels, n):
            _metrics.append(m)
    return _metrics, _clusters


def generate_DBSCAN_clusters(_data, _valleys_dists, _true_labels):
    _metrics = []
    _clusters = []
    i = 0
    for v in _valleys_dists:
        i += 1
        _clusters = DBSCAN(eps=v).fit_predict(_data)
        for m in cluster_eval(_data, _clusters, _true_labels, v):
            _metrics.append(m)
    return _metrics, _clusters


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
    create_dir("data/reduced/" + str(N))
    create_dir("data/restored/" + str(N))
    create_dir("data/reduced/" + str(N) + "/images")
    create_dir("data/restored/" + str(N) + "/images")
    save_images("data/reduced/" + str(N) + "/images/", _reduced)
    save_images("data/restored/" + str(N) + "/images/", _restored)
    return _reduced, _restored


def extract_features(_images):
    return np.column_stack([PCA(n_components=6).fit_transform(_images),
                            TSNE(n_components=6, method='exact').fit_transform(_images),
                            Isomap(n_components=6).fit_transform(_images)])


def get_original_feats_data(_images):
    if path.exists("data/original/original_feats.p"):
        print("Loading features...")
        _original_feats = pickle.load(open("data/original/original_feats.p", "rb"))
        print("Loaded original features...")
    else:
        print("Extracting features...")
        _original_feats = extract_features(_images)
        create_dir("data/original/")
        pickle.dump(_original_feats, open("data/original/original_feats.p", "wb"))
        print("Extracted original features...")
    return _original_feats


def get_reduced_feats_data(_images, N=18):
    if path.exists("data/reduced/" + str(N) + "/reduced_feats.p"):
        print("Loading features...")
        _reduced_feats = pickle.load(open("data/reduced/" + str(N) + "/reduced_feats.p", "rb"))
        print("Loaded reduced features...")
        _restored_feats = pickle.load(open("data/restored/" + str(N) + "/restored_feats.p", "rb"))
        print("Loaded restored features...")
    else:
        print("Reducing images...")
        _reduced_images, _restored_images = agglomerate_images_pixels(_images, N)
        _reduced_feats = extract_features(_reduced_images)
        pickle.dump(_reduced_feats, open("data/reduced/" + str(N) + "/reduced_feats.p", "wb"))
        print("Extracted reduced features...")
        _restored_feats = extract_features(_restored_images)
        pickle.dump(_restored_feats, open("data/restored/" + str(N) + "/restored_feats.p", "wb"))
        print("Extracted restored features...")
    return _reduced_feats, _restored_feats


def sequential_backward_elimination(_data, _true_labels, _classifier, _heuristic, _total_feats=18):
    _clusters = _classifier.fit_predict(_data)
    CURRENT_SELECTED_FEATS = filterKBest(_data, f_test(_data.iloc[LABELED], _true_labels), _total_feats)
    FINAL_SELECTED_FEATS = []
    _metrics = []
    best = -1
    for _n in range(0, _total_feats - 1):
        _selected_feats = _data.iloc[:, CURRENT_SELECTED_FEATS]
        _clusters = _classifier.fit_predict(_selected_feats)
        _m_results = cluster_eval(_selected_feats, _clusters, _true_labels, _total_feats - _n)
        found = _heuristic.get_score(_m_results)
        if found >= best:
            best = found
            FINAL_SELECTED_FEATS = CURRENT_SELECTED_FEATS
        for _m in _m_results:
            _metrics.append(_m)
        CURRENT_SELECTED_FEATS = filterKBest(_selected_feats, f_test(_selected_feats.iloc[LABELED], _true_labels),
                                             _total_feats - (_n + 1))
    return FINAL_SELECTED_FEATS, _metrics


class MyHeuristic:
    def __init__(self, evaluation_parameters, weights):
        self.evaluation_parameters = evaluation_parameters
        self.weights = weights

    def get_score(self, metrics):
        final_score = 0
        i = 0
        for p in self.evaluation_parameters:
            final_score += metrics[p][METRIC_VALUE] * self.weights[i]
            i += 1
        return final_score


class MyBisectingKMeans:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.kmeans = KMeans(n_clusters=2)

    def __fit_predict__(self, _data):
        lists_results = []
        indexes = [[]]

        def sort_by_length(l):
            return len(l)

        for example in range(_data.shape[0]):
            lists_results.append([])
            indexes[0].append(example)
        cluster_to_divide = 0

        # indexes = [[0,1,2,3,4,5,6,7,8,9], [...], ...]
        for example in range(self.n_iter):
            predict = self.kmeans.fit_predict(_data.iloc[indexes[cluster_to_divide]])

            new_cluster0 = np.array(indexes[cluster_to_divide])[np.where(predict == 0)[0]]
            new_cluster1 = np.array(indexes[cluster_to_divide])[np.where(predict == 1)[0]]

            for labeled_0 in new_cluster0:
                lists_results[labeled_0].append(0)
            for labeled_1 in new_cluster1:
                lists_results[labeled_1].append(1)

            indexes.pop(cluster_to_divide)
            indexes.append(list(new_cluster0))
            indexes.append(list(new_cluster1))
            indexes.sort(key=sort_by_length, reverse=True)
        self.list_result = lists_results
        cluster_results = np.zeros((_data.shape[0],))
        n_cluster = 0
        for cluster in indexes:
            cluster_results[np.array(cluster)] = n_cluster
            n_cluster += 1

        self.cluster_results = cluster_results

    def fit_predict(self, _data):
        self.__fit_predict__(_data)
        return self.cluster_results

    def fit_predict_report(self, _data):
        self.__fit_predict__(_data)
        return self.list_result


def experiment(_name, _feats, _labels, feature_selection=False, corr_filter=False, cluster_iter=0):
    print(150 * "_")
    # ________________________________________________________________INIT____________________________________________________________________________
    DATA_COLS = [f'f_{num}' for num in range(_feats.shape[1])]
    DATA_COLS.append("class")
    METRICS_COLS = ["metric", "x", "y"]
    data_df = pd.DataFrame(np.column_stack([_feats, _labels[:, -1]]), columns=DATA_COLS)
    kmeans_data_df = data_df
    dbscan_data_df = data_df
    bisec_data_df = data_df
    agglomerative_data_df = data_df
    if feature_selection:
        if corr_filter:
            _name = "SELECTED_LOW_" + _name
        else:
            _name = "SELECTED_" + _name
    elif corr_filter:
        _name = "LOW_" + _name
    # ________________________________________________________________________________________________________________________________________________
    # ________________________________________________________________________________________________________________________________________________

    # Finding best eps
    _5dists = np.sort(np.sort(np.linalg.norm(_feats - _feats[:, None], axis=-1), axis=-1)[::-1][:, 4])[::-1]
    valleys = find_peaks_cwt(_5dists * (-1), np.arange(1, 20))
    valleys_dists = _5dists[valleys]
    best_eps = valleys_dists[1]
    # Plotting 5 dists with best eps found
    plot_5dist(_5dists, valleys, "dbscan/" + _name + "_5-dists", "5 Distances")

    # ________________________________________________________________________________________________________________________________________________
    # -----------------------------------------------------------FEATURE SELECTION--------------------------------------------------------------------

    # ----------------------------------------------------SEQUENTIAL BACKWARD ELIMINATION-------------------------------------------------------------
    if feature_selection:
        heuristic = MyHeuristic(HEURISTIC_VALUES, HEURISTIC_WEIGHTS)
        KMEANS_SELECTED, kmeans_metrics = sequential_backward_elimination(data_df, _labels[LABELED][:, 1],
                                                                          KMeans(n_clusters=3), heuristic)
        DBSCAN_SELECTED, dbscan_metrics = sequential_backward_elimination(data_df, _labels[LABELED][:, 1],
                                                                          DBSCAN(eps=best_eps), heuristic)
        BISECT_SELECTED, bisec_metrics = sequential_backward_elimination(data_df, _labels[LABELED][:, 1],
                                                                         MyBisectingKMeans(n_iter=3), heuristic)
        AGGLOMERATIVE_SELECTED, agglomerative_metrics = sequential_backward_elimination(data_df, _labels[LABELED][:, 1],
                                                                                        AgglomerativeClustering(n_clusters=3),
                                                                                        heuristic)
        print(str(KMEANS_SELECTED) + "," + str(len(KMEANS_SELECTED)))
        print(str(DBSCAN_SELECTED) + "," + str(len(DBSCAN_SELECTED)))
        print(str(BISECT_SELECTED) + "," + str(len(BISECT_SELECTED)))
        print(str(AGGLOMERATIVE_SELECTED) + "," + str(len(AGGLOMERATIVE_SELECTED)))
        kmeans_data_df = data_df.iloc[:, KMEANS_SELECTED]
        dbscan_data_df = data_df.iloc[:, DBSCAN_SELECTED]
        bisec_data_df = data_df.iloc[:, BISECT_SELECTED]
        agglomerative_data_df = data_df.iloc[:, AGGLOMERATIVE_SELECTED]
        # Plotting cluster metrics as a function of the number of the remaining best features
        plot_cluster_line_metrics(pd.DataFrame(kmeans_metrics, columns=METRICS_COLS),
                                  "kmeans/" + _name + "_kmeans_features_metrics", "Features",
                                  "KMeans Features Metrics")
        plot_cluster_line_metrics(pd.DataFrame(dbscan_metrics, columns=METRICS_COLS),
                                  "dbscan/" + _name + "_dbscan_features_metrics", "Features",
                                  "DBSCAN Features Metrics")
        plot_cluster_line_metrics(pd.DataFrame(bisec_metrics, columns=METRICS_COLS),
                                  "bisecting/" + _name + "_bisecting_features_metrics", "Features",
                                  "Bisecting KMeans Features Metrics")
        plot_cluster_line_metrics(pd.DataFrame(agglomerative_metrics, columns=METRICS_COLS),
                                  "agglomerative/" + _name + "_agglomerative_features_metrics", "Features",
                                  "Agglomerative Features Metrics")

    # -----------------------------------------------------------CORRELATION FILTERING----------------------------------------------------------------
    if corr_filter:
        data_df = data_df.iloc[:, filter_low_high_corr(data_df.iloc[:, :-1].corr())[0]]
        if feature_selection:
            kmeans_data_df = kmeans_data_df.iloc[:, filter_low_high_corr(kmeans_data_df.corr())[0]]
            dbscan_data_df = dbscan_data_df.iloc[:, filter_low_high_corr(dbscan_data_df.corr())[0]]
            bisec_data_df = bisec_data_df.iloc[:, filter_low_high_corr(bisec_data_df.corr())[0]]
            agglomerative_data_df = bisec_data_df.iloc[:, filter_low_high_corr(agglomerative_data_df.corr())[0]]
        else:
            kmeans_data_df = data_df
            dbscan_data_df = data_df
            bisec_data_df = data_df
            agglomerative_data_df = data_df

    # Plotting features' correlation heatmap
    plot_heatmap(data_df.iloc[:, :-1], _name + "_full_heatmap", "Features Heatmap")
    if feature_selection:
        plot_heatmap(kmeans_data_df, _name + "_kmeans_feats_heatmap", "Features Heatmap")
        plot_heatmap(dbscan_data_df, _name + "_dbscan_feats_heatmap", "Features Heatmap")
        plot_heatmap(bisec_data_df, _name + "_bisecting_feats_heatmap", "Features Heatmap")
        plot_heatmap(agglomerative_data_df, _name + "_agglomerative_feats_heatmap", "Features Heatmap")

    # ________________________________________________________________________________________________________________________________________________
    # -----------------------------------------------------------CLUSTER GENERATION-------------------------------------------------------------------
    # Generating clusters with the best estimated parameters
    _kmeans_clusters = KMeans(n_clusters=3).fit_predict(kmeans_data_df)
    _dbscan_clusters = DBSCAN(eps=best_eps).fit_predict(dbscan_data_df)
    _bisecting_report_clusters = MyBisectingKMeans(n_iter=2).fit_predict_report(bisec_data_df)
    _bisecting_clusters = MyBisectingKMeans(n_iter=2).fit_predict(bisec_data_df)
    _agglomerative_clusters = AgglomerativeClustering(n_clusters=3).fit_predict(agglomerative_data_df)

    kmeans_best_2_dims = filterKBest(kmeans_data_df, f_test(kmeans_data_df, _kmeans_clusters), 2)
    dbscan_best_2_dims = filterKBest(dbscan_data_df, f_test(dbscan_data_df, _dbscan_clusters), 2)
    bisecting_best_2_dims = filterKBest(bisec_data_df, f_test(bisec_data_df, _bisecting_clusters), 2)
    agglomerative_best_2_dims = filterKBest(agglomerative_data_df, f_test(agglomerative_data_df, _agglomerative_clusters), 2)

    _kmeans_plt_data = pd.concat(
        [kmeans_data_df.iloc[:, kmeans_best_2_dims], pd.Series(_kmeans_clusters, name='class')], axis=1)
    _dbscan_plt_data = pd.concat(
        [dbscan_data_df.iloc[:, dbscan_best_2_dims], pd.Series(_dbscan_clusters, name='class')], axis=1)
    _bisecting_plt_data = pd.concat(
        [bisec_data_df.iloc[:, bisecting_best_2_dims], pd.Series(_bisecting_clusters, name='class')], axis=1)
    _agglomerative_plt_data = pd.concat(
        [agglomerative_data_df.iloc[:, agglomerative_best_2_dims], pd.Series(_agglomerative_clusters, name='class')], axis=1)

    # Plotting clusters with dimensionality reduction (keeping the two features with the highest f-score)
    plot_joint_plot(_kmeans_plt_data, "kmeans/" + _name + "_clusters", "Kmeans")
    plot_joint_plot(_dbscan_plt_data, "dbscan/" + _name + "_clusters", "DBSCAN")
    plot_joint_plot(_bisecting_plt_data, "bisecting/" + _name + "_clusters", "Bisecting KMeans")
    plot_joint_plot(_agglomerative_plt_data, "agglomerative/" + _name + "_clusters", "Agglomerative")
    # Generating cluster reports
    utils.report_clusters(_labels[:, 0], _kmeans_clusters, "clusters/" + _name + "_kmeans_clusters.html")
    utils.report_clusters(_labels[:, 0], _dbscan_clusters, "clusters/" + _name + "_dbscan_clusters.html")
    utils.report_clusters_hierarchical(_labels[:, 0], _bisecting_report_clusters,
                                       "clusters/" + _name + "_bisecting_kmeans_clusters.html")
    utils.report_clusters(_labels[:, 0], _agglomerative_clusters, "clusters/" + _name + "_agglomerative_clusters.html")

    # ________________________________________________________________________________________________________________________________________________
    # ------------------------------------------------------Cluster Parameter Evaluation--------------------------------------------------------------
    if cluster_iter > 0:
        # KMeans
        kmeans_metrics_df = pd.DataFrame(
            generate_KMeans_clusters(kmeans_data_df, cluster_iter, labels[LABELED][:, 1])[0],
            columns=METRICS_COLS)
        # DBSCAN
        eps_range = []
        for num in np.arange(0, best_eps, best_eps / cluster_iter):
            eps_range.append(best_eps - num)
        eps_range.reverse()
        dbscan_metrics_df = pd.DataFrame(generate_DBSCAN_clusters(dbscan_data_df, eps_range, labels[LABELED][:, 1])[0],
                                         columns=METRICS_COLS)
        # Bisecting K-Means
        bisecting_metrics_df = pd.DataFrame(
            generate_Bisecting_KMeans_clusters(bisec_data_df, cluster_iter, labels[LABELED][:, 1])[0],
            columns=METRICS_COLS)
        # agglomerative
        agglomerative_metrics_df = pd.DataFrame(
            generate_Agglomerative_clusters(agglomerative_data_df, cluster_iter, labels[LABELED][:, 1])[0],
            columns=METRICS_COLS)

        plot_cluster_line_metrics(kmeans_metrics_df, "/kmeans/" + _name + "_kmeans_parameter_metrics", "Clusters",
                                  "KMeans Cluster Metrics")
        plot_cluster_line_metrics(dbscan_metrics_df, "/dbscan/" + _name + "_dbscan_parameter_metrics", "ε",
                                  "DBSCAN Cluster Metrics")
        plot_cluster_line_metrics(bisecting_metrics_df, "/bisecting/" + _name + "_bisecting_KMeans_parameter_metrics", "Iterations",
                                  "Bisecting KMeans Cluster Metrics")
        plot_cluster_line_metrics(agglomerative_metrics_df, "/agglomerative/" + _name + "_agglomerative_parameter_metrics", "Clusters",
                                  "Agglomerative Cluster Metrics")
    # ________________________________________________________________________________________________________________________________________________


SILHOUETTE_SCORE = 0
PRECISION_SCORE = 1
RECALL = 2
F1_MEASURE = 3
ADJUSTED_RANDOM_INDEX = 4
RANDOM_INDEX = 5
METRIC_VALUE = 2
HEURISTIC_VALUES = [SILHOUETTE_SCORE, F1_MEASURE, ADJUSTED_RANDOM_INDEX]
HEURISTIC_WEIGHTS = [3, 3, 1]
labels = np.loadtxt("labels.txt", delimiter=",")
LABELED = np.where(labels[:, -1] != 0)
images = utils.images_as_matrix()
create_dir("data")
create_dir("plots")
create_dir("plots/kmeans")
create_dir("plots/dbscan")
create_dir("plots/bisecting")
create_dir("plots/agglomerative")
create_dir("clusters")

original_feats = get_original_feats_data(images)
experiment("original", original_feats, labels, feature_selection=True, corr_filter=True, cluster_iter=10)
# experiment("standardized", standardize(original_feats), labels, feature_selection=True, corr_filter=False, cluster_iter=10)
# experiment("normalized", normalize(original_feats), labels, feature_selection=True, corr_filter=True, cluster_iter=10)
"""
for n in [10, 50, 250]:
    reduced_feats, restored_feats = get_reduced_feats_data(images, n)
    experiment(str(n) + "_reduced", reduced_feats, labels, feature_selection=True, corr_filter=True, cluster_iter=10)
    experiment(str(n) + "_restored", restored_feats, labels, feature_selection=True, corr_filter=True, cluster_iter=10)
"""
