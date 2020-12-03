import tp2_aux as utils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)
pca = PCA(n_components=6)
tsne = TSNE(n_components=6, method='exact')
iso = Isomap(n_components=6)

# Convert images to panda
img_mat = utils.images_as_matrix()
# Extract features
print("Extracting features...")
feats = pd.DataFrame(np.column_stack([[
    "PCA-1", "PCA-2", "PCA-3", "PCA-4", "PCA-5", "PCA-6",
    "TSNE-1", "TSNE-2", "TSNE-3", "TSNE-4", "TSNE-5", "TSNE-6",
    "ISO-1", "ISO-2", "ISO-3", "ISO-4", "ISO-5", "ISO-6",
], pca.fit_transform(img_mat), tsne.fit_transform(img_mat), iso.fit_transform(img_mat)]))
print("Features extracted.")
# Create clusters


# Plot clusters to features


# Evaluate clusters


# Plot evaluation metrics
