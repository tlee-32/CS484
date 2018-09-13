import numpy as np
from sklearn.model_selection import KFold
from gensim.models.doc2vec import TaggedDocument
from sklearn.neighbors import KNeighborsClassifier
from knn import *

def findOptimalKForKNN(trainingData):
    folds = createKFolds(np.array(trainingData), kFolds=7)
    knnResults = kFoldCV(folds, 3)
    #print(knnResults)

"""
    Iterates through each fold, performs knn and outputs the number of correct
    and incorrect results for each fold

    return - list of final knn results
"""
def kFoldCV(folds, initialK):
    k = initialK
    results = {}
    c = 0
    # classify each fold from k=1 to k=len(folds)
    for (trainData, testData) in folds:
        # Train the model using the training sets
        knn = KNNClassifier(k)
        knn.fit(trainData, retrain=True) # retrain each time
        correct, incorrect = 0, 0
        print(k)
        for document in testData: # testData is smaller than 18506
            predictionValue = knn.classify(document.words)
            actualValue = document.tags[0]
            if(predictionValue == actualValue[:2]):
                correct += 1
            else:
                incorrect += 1
        key = 'k%d' % k
        results[key] = (correct, incorrect, len(testData))
        print(results)
        k += 1

    return results

"""
    Split training and test data using cross validation.

    return - k-folds
"""
def createKFolds(trainingData, kFolds=3):
    kf = KFold(n_splits=kFolds, shuffle=True, random_state=1)
    folds = []
    testData = []
    for train, test in kf.split(trainingData):
        # convert to TaggedDocument in order to fit the data
        trainData = [TaggedDocument(arr[0], arr[1]) for arr in trainingData[train]]
        testData = [TaggedDocument(arr[0], arr[1]) for arr in trainingData[test]]
        folds.append((trainData, testData))
    return folds
