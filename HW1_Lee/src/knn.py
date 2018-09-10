import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class KNNClassifier:

    def __init__(self, k):
        self.model = None
        self.k = k
    """
        Classifies the k-nearest-neighbors for the given document.

        return - classified result
    """
    def classify(self, document):
        documentVector = self.model.infer_vector(document) # vectorizes unseen test document
        trainingVectors = self.model.docvecs.vectors_docs # all document vectors of training data
        nearestNeighbors = findKNearestNeighbors(self.k, trainingVectors, documentVector)

        # TODO: MAJORITY VOTE

        #print(model.docvecs.index_to_doctag(0)) #find doctag
        #print(model.docvecs.doctags)
        #print(model.docvecs.doctag_syn0[0]) #find vector
        knn = findKNearestNeighbors(k, trainingVectors, documentVector)
        print(trainingVectors[knn[0][0]])
        #print(trainDocs[knn[0][0]])
        print(trainingVectors[knn[1][0]])
        #print(trainDocs[knn[1][0]])
        print(trainingVectors[knn[2][0]])
        #print(trainDocs[knn[2][0]])

        #return result

    """
        Fits the data to the knn model using Doc2Vec.
    """
    def fit(self, documents):
        self.model = trainDoc2VecModel(documents, vectorSize=50, window=2, minCount=2, epochs=5)

    """
        Finds the k-nearest neighbors in a 2D array holding vectors given a 1D vector.
    """
    def findKNearestNeighbors(self, trainingVectors, vector):
        distances = euclideanDistance(trainingVectors, vector, isMatrixNorm=True)
        nearestNeighbors = findKSmallestValues(self.k, distances)
        return nearestNeighbors

    """
        Partition sorts the given array to grab the k-smallest values in an array.
    """
    def findKSmallestValues(self, arr):
        indices = np.argpartition(arr, self.k)[:self.k] # indices to k-smallest values in arr
        idxValueTuples = [(idx, val) for (idx, val) in zip(indices, arr[indices])]
        return idxValueTuples

    """
        Computes euclidean distance between 1D/2D vectors by vector/matrix norm calculation.
        Vector/matrix norm is simply the sqrt(sum of the squared elements)
        in a vector.

        isMatrixNorm - determines if u or v is 2D
    """
    def euclideanDistance(u, v, isMatrixNorm=False):
        axis = 1 if isMatrixNorm else 0 # decides if 1D or 2D axes computation is needed
        vectorDiffs = v - u
        return np.linalg.norm(vectorDiffs, axis=axis)
