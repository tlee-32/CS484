import numpy as np
import random
from scipy.sparse import isspmatrix
from .helper.dist import getNewLabels, getNewCentroids # pylint: disable=E0402

class KMeansClassifier:
    def __init__(self, clusters, metric):
        self.clusters = clusters
        self.metric = metric

    def fit(self, data):
        centroids = self.getInitialCentroids(data=data)
        return self.kMeans(data, centroids)

    def kMeans(self, data, initialCentroids):
        converged = False
        newCentroids = initialCentroids
        labels = []
        
        while not converged:
            # assign labels
            labels = getNewLabels(data, newCentroids, metric=self.metric)
            
            # compute new centroids
            oldCentroids = newCentroids
            newCentroids = getNewCentroids(data, labels, numCentroids=self.clusters)

            # check if centroids have changed
            converged = np.array_equal(oldCentroids, newCentroids)
        return labels
    
    def getInitialCentroids(self, data):
        rowIdxLimit = 0
        # obtain row limit depending on sparse or non-sparse matrix
        if isspmatrix(data):
            rowIdxLimit = data.shape[0]
        else:
            rowIdxLimit = len(data)
        # choose UNIQUE random indices for centroids
        randIndices = np.random.choice(rowIdxLimit, self.clusters, replace=False)
        centroids = data[randIndices]
        return centroids
