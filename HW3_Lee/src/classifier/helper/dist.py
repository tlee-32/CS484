import numpy as np
from sklearn.metrics import pairwise_distances

"""
    Assigns an index to the centroid list based on the closest point.
    Example: an index of 1 for point 0 = point 0 is assigned to centroid 1

    return - newly assigned labels (or indices)
"""
def getNewLabels(points, centroids, metric='euclidean'):
    # compute distances by specified metric
    distances = pairwise_distances(X=points, Y=centroids, metric=metric)
    # grab the indices of the closest centroids for each point
    labels = distances.argmin(axis=1)
    return labels

"""
    Compute new centroids by taking the mean of newly labelled points
    for each centroid.
"""
def getNewCentroids(points, labels, numCentroids):
    newCentroids = []
    # compute mean for each centroid
    for centroid in range(numCentroids):
        # indices to the points based on current centroid
        pointIndicesOfCurrentCentroid = labels == centroid
        currentCentroidPoints = points[pointIndicesOfCurrentCentroid]
        # compute mean for each element in 2-D array or sparse matrix
        newCentroidMeanMatrix = np.mean(currentCentroidPoints, axis=0)
        newCentroidMean = np.ravel(newCentroidMeanMatrix)
        newCentroids.append(newCentroidMean)
    return newCentroids