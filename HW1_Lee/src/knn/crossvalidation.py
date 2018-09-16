import numpy as np
from sklearn.model_selection import KFold
from gensim.models.doc2vec import TaggedDocument
from .knn import KNNClassifier

"""
    Finds the optimal k for k-nearest neighbors classifier. The optimal value
    of k is chosen by the highest accuracy between the folds.

    return - optimal k
"""
def findOptimalKForKNN(trainingData):
    folds = createKFolds(np.array(trainingData), kFolds=3)
    knnResults = kFoldCV(folds, initialK=5)
    optimalK = max(knnResults, key=knnResults.get) # retrieve highest accuracy from dict, knnResults
    return optimalK

"""
    Iterates through each fold, performs knn and outputs the number of correct
    and incorrect results for each fold.

    return - list of kNN accuracy for given k-folds
"""
def kFoldCV(folds, initialK):
    k = initialK
    results = {}
    # classify each fold from k=initialK to k=len(folds)
    for (trainData, testData) in folds:
        knn = KNNClassifier(k) # train the model using the training sets
        knn.fit(trainData, retrain=True) # retrain Doc2Vec model for each fold
        correct, incorrect = 0, 0
        print(k)
        for document in testData: # testData is smaller than 18506
            predictionValue = knn.classify(document.words)
            actualValue = document.tags[0]
            if(predictionValue == actualValue[:2]):
                correct += 1
            else:
                incorrect += 1
        totalTestData = len(testData)
        accuracy = correct / totalTestData
        results[k] = accuracy # map accuracy to value of k
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
    for train, test in kf.split(trainingData):
        # convert to TaggedDocument in order to fit the data
        trainData = [TaggedDocument(arr[0], arr[1]) for arr in trainingData[train]]
        testData = [TaggedDocument(arr[0], arr[1]) for arr in trainingData[test]]
        folds.append((trainData, testData))
    return folds
