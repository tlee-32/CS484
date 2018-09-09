# Holds functions to vectorize documents using gensim

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

def trainDoc2VecModel(trainDocs):
    model = Doc2Vec(trainDocs, vector_size=50, window=2, min_count=2, workers=4, epochs=5)
    model.train(trainDocs, total_examples=model.corpus_count, epochs=model.epochs)
    test = ['perfect', 'new', 'parents', 'able', 'keep', 'track', 'babys', 'feeding', 'sleep', 'diaper', 'change', 'schedule', 'first', 'two', 'half', 'months', 'life',
'made', 'life', 'easier', 'doctor', 'would', 'ask', 'questions', 'habits', 'right']
    vec1 = model.infer_vector(test) # vectorizes test data
    small = 100
    a = 0

    findKNearestNeighbors(3, np.array(model.docvecs), vec1)
    #print(small)
    #print(list(trainDocs)[a])

"""
    knn.py
    Finds the k-nearest neighbors using partitioning
"""
def findKNearestNeighbors(k, trainingVectors, vector):
    neighbors = []

    distances = euclideanDistance(trainingVectors, vector, isMatrixNorm=True)
    nearestNeighbors = findKSmallestValues(k, distances)

"""
    Utilizes partition sorting to grab the k-smallest values in an array.
"""
def findKSmallestValues(k, arr):
    indices = np.argpartition(arr, k)[:k] # indices to k-smallest values in arr
    return arr[indices]

"""
    Computes euclidean distance between 1D/2D vectors by vector/matrix norm calculation.
    Vector/matrix norm is simply the sqrt(sum of the squared elements)
    in a vector.

    isMatrixNorm - determines if u or v is 2D
"""
def euclideanDistance(u, v, isMatrixNorm=False):
    axis = 1 if isMatrixNorm else 0
    vectorDiffs = v - u
    return np.linalg.norm(vectorDiffs, axis=axis)
