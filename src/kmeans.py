from matplotlib.colors import rgb2hex
from sklearn import cluster, decomposition, metrics
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.cm as cm


def applyKmeans(dataset, n_iterations, n_clusters=False):
    pca_ = decomposition.PCA(n_components=2)
    X_pca = pca_.fit_transform(dataset)

    if not n_clusters:
        n_clusters = silhouette(X_pca)

    fig, axes = plt.subplots(math.ceil(n_iterations / 3), min(n_iterations, 3))
    fig.set_figheight(3 * math.ceil(n_iterations / 3))
    fig.set_figwidth(3 * min(n_iterations, 3))
    for i in range(n_iterations):
        X_pca = it_kmeans(n_clusters, X_pca, pca_, i, axes)
    plt.draw()


def it_kmeans(n_clusters, X_pca, pca_, ii, axes):
    cl = cluster.KMeans(n_clusters=n_clusters)
    X_pca = cl.fit_transform(X_pca)

    new_labels = cl.labels_
    centers = cl.cluster_centers_
    kmeans_labels = ['Cluster ' + str(i) for i in range(1, n_clusters + 1)]
    explained_var = np.round(pca_.explained_variance_ratio_ * 100, decimals=2)

    colors = cm.rainbow(np.linspace(0, 1, len(kmeans_labels)))
    row = int(ii / 3)
    col = ii % 3

    for i in range(n_clusters):
        axes[row, col].scatter(X_pca[new_labels == i, 0], X_pca[new_labels == i, 1], c=rgb2hex(colors[i]),
                               label=kmeans_labels[i])
    axes[row, col].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8)
    axes[row, col].set_xlabel('PC1 - {0}%'.format(explained_var[0]))
    axes[row, col].set_ylabel('PC2 - {0}%'.format(explained_var[1]))

    return X_pca


def silhouette(X_pca, print_=False):
    kmeans_per_k = [cluster.KMeans(n_clusters=k).fit(X_pca) for k in range(1, 10)]
    silhouette_scores = [metrics.silhouette_score(X_pca, kmeans.labels_) for kmeans in kmeans_per_k[1:]]
    k = np.argmax(silhouette_scores) + 2
    if print_:
        plt.plot(range(2, 10), silhouette_scores, "bo-", color="blue")
        plt.scatter(k, silhouette_scores[k - 2], c='red', s=400)
    return k
