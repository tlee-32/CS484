import numpy as np
from vectorizermodel import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity


class KNNClassifier:

    def __init__(self, k):
        self.model = None
        self.k = k
    """
        Classifies the k-nearest-neighbors for the given document.

        return - classified result
    """
    def classify(self, document):
        documentVector = self.model.vectorizeDocument(document) # vectorizes unseen test document
        trainingVectors = self.model.getTrainingVectors() # all document vectors of training data
        nearestNeighbors = self.findKNearestNeighbors(trainingVectors, documentVector)
        nearestNeighbors = self.removeSuffixFromTags(nearestNeighbors)# values should only be labelled with '+1' or '-1' prefix
        majorityVote = self.getWeightedMajorityVote(nearestNeighbors)
        return majorityVote

    """
        Fits the data to the knn model using the Doc2Vec.
    """
    def fit(self, documents, retrain=True):
        self.model = Doc2VecModel(trainDocs=documents, vectorSize=100, window=3, minCount=1, epochs=3, retrain=retrain)

    """
        Finds the k-nearest neighbors in a 2D array holding vectors given a 1D vector.
    """
    def findKNearestNeighbors(self, trainingVectors, vector):
        distances = self.euclideanDistance(trainingVectors, vector, isMatrixNorm=True)
        #distances = self.cosineSimilarity(trainingVectors, vector, isMatrixNorm=True)
        nearestNeighbors = self.findKSmallestValues(distances)
        return nearestNeighbors

    """
        Partition sorts the given array to grab the k-smallest values in an array.
    """
    def findKSmallestValues(self, arr):
        indices = np.argpartition(arr, self.k)[:self.k] # indices to k-smallest values in arr
        idxValueTuples = [(idx, val) for (idx, val) in zip(indices, arr[indices])]
        return idxValueTuples

    """
        Calculates the weighted majority vote using the inverse of the distance.
    """
    def getWeightedMajorityVote(self, nearestNeighbors):
        weightedClasses = {name: 0 for (name, dist) in nearestNeighbors} # create mapping of class names and their weights
        # calculate weights for classes
        for (idx, dist) in nearestNeighbors:
            weightedVote = 1 / dist # inverse of distance allows closer points to have more weight
            weightedClasses[idx] += weightedVote # sum of weightedVotes for a class
        # determine winner
        winner = max(weightedClasses, key=weightedClasses.get)
        return winner

    """
        Substring the tag from TaggedDocument such that it only holds the class label.
    """
    def removeSuffixFromTags(self, classValueList):
        result = []
        for tag, dist in classValueList:
            name = self.model.findDocTag(tag)
            result.append((name[:2], dist))
        return result

    """
        Computes euclidean distance between 1D/2D vectors by vector/matrix norm calculation.
        Vector/matrix norm is simply the sqrt(sum of the squared elements)
        in a vector.

        isMatrixNorm - determines if u or v is 2D
    """
    def euclideanDistance(self, u, v, isMatrixNorm=False):
        axis = 1 if isMatrixNorm else 0 # decides if 1D or 2D axes computation is needed
        vectorDiffs = v - u
        return np.linalg.norm(vectorDiffs, axis=axis)

    def cosineSimilarity(self, u, v, isMatrixNorm=False):
        axis = 1 if isMatrixNorm else 0 # decides if 1D or 2D axes computation is needed
        return cosine_similarity(u, v.reshape(1, -1)).flatten()
