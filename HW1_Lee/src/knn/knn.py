import numpy as np
from feature.doc2vecmodel import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import utils

"""
    Implementation of k-Nearest Neighbors
"""
class KNNClassifier:
    def __init__(self, k):
        self.model = None
        self.k = k
    """
        Classifies the k-NN for the given document.

        return - winner of k-NN
    """
    def classify(self, document):
        documentVector = self.model.vectorizeDocument(document) # vectorizes unseen test document
        trainingVectors = self.model.getTrainingVectors() # all document vectors of training data
        nearestNeighbors = self.findKNearestNeighbors(trainingVectors, documentVector)
        nearestNeighbors = self.removeSuffixFromTags(nearestNeighbors) # values should only be labelled with '+1' or '-1' prefix
        majorityVote = self.getWeightedMajorityVote(nearestNeighbors)
        return majorityVote

    """
        Fits the data to the knn model using the Doc2Vec.
    """
    def fit(self, documents, retrain=True):
        self.model = Doc2VecModel(trainDocs=utils.shuffle(documents), vectorSize=100, window=10, minCount=3, epochs=20, retrain=retrain)

    """
        Finds the k-nearest neighbors in a 2D array holding vectors given a 1D vector.
    """
    def findKNearestNeighbors(self, trainingVectors, vector):
        distances = self.euclideanDistance(trainingVectors, vector)
        nearestNeighbors = self.findKSmallestValues(distances)
        return nearestNeighbors

    """
        Partition sorts the given array to grab the k-smallest values in an array.

        return - k-smallest values
    """
    def findKSmallestValues(self, arr):
        indices = np.argpartition(arr, self.k)[:self.k] # indices to k-smallest values in arr
        idxValueTuples = [(idx, val) for (idx, val) in zip(indices, arr[indices])]
        return idxValueTuples

    """
        Calculates the weighted majority vote using the inverse of the distance.

        return - weighted majority vote winner
    """
    def getWeightedMajorityVote(self, nearestNeighbors):
        weightedClasses = {name: 0 for (name, dist) in nearestNeighbors} # create mapping of class names and their weights
        # calculate weights for classes
        for (idx, dist) in nearestNeighbors:
            weightedVote = 1 / dist # inverse of distance allows closer points to have more weight
            weightedClasses[idx] += weightedVote # sum of weightedVotes for a class
        winner = max(weightedClasses, key=weightedClasses.get) # determine winner
        return winner

    """
        Substring the tag from TaggedDocument such that it only holds the class label.

        return - '+1' or '-1'
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

        return - euclidean distances of all vectors
    """
    def euclideanDistance(self, u, v):
        vectorDiffs = v - u
        return np.linalg.norm(vectorDiffs, axis=1)
