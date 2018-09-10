import numpy as np
from sklearn.model_selection import KFold
from gensim.models.doc2vec import TaggedDocument

def findOptimalKForKNN(trainingData):
    folds = createKFolds(np.array(trainingData), kFolds=10)
    #knnResults = kFoldCV(folds)

"""
    Iterates through each fold and performs knn.

    return - list of final knn results
"""
def kFoldCV(folds):
    k = 1
    results = []
    # classify each fold from k=1 to k=len(folds)
    for (trainData, testData) in folds:
        knn = KNNClassifier(k)
        knn.fit(trainData)
        for document in testData:
            predictionValue = knn.classify(document.words),
            actualValue = document.tags[0]
            results.append((predicitionValue, actualValue)))
        k += 1

    return results

"""
    Split training and test data using cross validation.

    return - k-folds
"""
def createKFolds(trainingData, kFolds=3):
    kf = KFold(n_splits=kFolds, shuffle=True, random_state=1)
    folds = []
    for train, test in kf.split(trainingData):
        # convert to TaggedDocument in order to fit the data
        trainingTag = [TaggedDocument(arr[0], arr[1]) for arr in trainingData[train]]
        testTag = [TaggedDocument(arr[0], arr[1]) for arr in trainingData[test]]

        folds.append((trainingTag, testTag))
    return folds
