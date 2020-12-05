import tp2_aux as utils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans

import numpy as np
import pickle
import pandas as pd
from functools import partial
from tqdm import tqdm
from os import path


def normalize(_feats):


def f_test(_feats, _labels):
    _f, probs = f_classif(_feats, _labels)
    return _f


np.set_printoptions(precision=4, suppress=True)
tqdm = partial(tqdm, position=0, leave=True)
labels = np.loadtxt("labels.txt", delimiter=",")
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


f_scores = f_test(feats[np.where(labels[:, -1] != 0)], labels[np.where(labels[:, -1] != 0)][:, -1])
print(f_scores)

# Create clusters

KMeans()



# Plot clusters to features


# Evaluate clusters


# Plot evaluation metrics
