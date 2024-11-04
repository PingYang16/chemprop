import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import numpy as np

df_fps = pd.read_csv("fps_rdkit_2d_0.csv")
df_mols = pd.read_csv("~/ALMS/data/c1_c20.csv")

pca = PCA(n_components = 3)
principalComp = pca.fit_transform(df_fps)

tsne = TSNE(n_components = 3)
tsneComp = tsne.fit_transform(df_fps)
np.save("tsneComp_rdkit.npy", tsneComp)

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(121, projection='3d')
ax.scatter(principalComp[:,0], principalComp[:,1], principalComp[:,2], c='r', marker='o')
ax.set_title('PCA 3D Plot')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(tsneComp[:,0], tsneComp[:,1], tsneComp[:,2], c='b', marker='^')
ax2.set_title('t-SNE 3D Plot')
ax2.set_xlabel('tSNE1')
ax2.set_ylabel('tSNE2')
ax2.set_zlabel('tSNE3')

plt.savefig('3d_embedding_plots_rdkit.png', dpi=300)

n_cluster = 50
kmeans = KMeans(n_clusters=n_cluster, random_state=42)
clusters = kmeans.fit_predict(tsneComp)

representative_indices = []
for cluster_id in range(n_cluster):
    cluster_indices = np.where(clusters == cluster_id)[0]
    centroid = kmeans.cluster_centers_[cluster_id]
    distances_to_centroid = np.linalg.norm(tsneComp[cluster_indices] - centroid, axis=1)
    closest_index = cluster_indices[np.argmin(distances_to_centroid)]
    representative_indices.append(closest_index)

top_50_mols = df_mols.iloc[representative_indices]
top_50_mols.to_csv('top_50_mols_rdkit.csv', index=False)