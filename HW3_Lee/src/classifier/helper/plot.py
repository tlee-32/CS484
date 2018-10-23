import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import nipy_spectral # pylint: disable=E0611
from sklearn.metrics import silhouette_samples, silhouette_score

def plotSilhouetteLine(kMeans, data, minK=3, maxK=21, step=2):
    points = []
    xAxis = [i for i in range(minK,maxK+1,step)]
    # Initialize x-axis 
    plt.xlim([minK, maxK])
    plt.xticks(xAxis)
    
    # Run K-Means for (minK+maxK)/2 iterations with K increasing by steps of 2 
    for nClusters in range(minK, maxK+1, step):
        kMeans.clusters = nClusters
        labels = kMeans.fit(data)
        score = silhouette_score(data, labels, metric='cosine')
        points.append(score)
    
    plt.title("Silhouette coefficients for 8 different K values in K-means clustering")
    plt.xlabel("Value of K")
    plt.ylabel("Silhouette coefficients")
    plt.plot(xAxis, points, linestyle='-', marker='o', color='b')
    plt.show()

"""
    Plots the silhouette scores for each cluster. 
    Implementation by Sklearn's K-Means Clustering Example.
        - http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
"""
def plotSilhouetteGraph(kMeans, data, dataSize, clusters, metric):
    # Initialize plot
    plt.xlim([-0.1, 1])
    plt.ylim([0, dataSize + (clusters + 1) * 10])

    # Perform K-Means cluster
    labels = kMeans.fit(data)

    # Compute the silhouette scores and average
    silhouetteAvg = silhouette_score(data, labels, metric=metric)
    silhouetteSampleValues = silhouette_samples(data, labels)
    
    y_lower = 10
    for i in range(clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouetteSampleValues[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = nipy_spectral(float(i) / clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouetteAvg, color="red", linestyle="--")
    
    plt.yticks([])
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

def plotSparsePCAExplainedVariance(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
