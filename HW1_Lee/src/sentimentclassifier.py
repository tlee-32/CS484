import datetime # TESTING
# Python 3
from filetokenizer import *
from crossvalidation import *
from knn import KNNClassifier
import smart_open
import numpy as np

def main():
    start = datetime.datetime.now()

    reviews = list(readReviews("../train.data", loadFile=True, isTrainingFile=True))[0]
    testReviews = list(readReviews("../test.data", loadFile=True, isTrainingFile=False))[0]
    #k = findOptimalKForKNN(reviews) # find optimal k using cross validation
    k = 7
    knn = KNNClassifier(k)
    knn.fit(reviews, retrain=False) # build Doc2Vec model of training data
    #print(knn.classify(test))
    sentiment = classifySentimentWithKNN(knn, testReviews)
    end = datetime.datetime.now()
    totalSeconds = (end - start).total_seconds()
    print("Duration: {}".format(totalSeconds))


def classifySentimentWithKNN(knn, testData):
    with smart_open.smart_open("../output.data", "w") as f:
        for review in testData:
            sentiment = knn.classify(review) # sentiment as +1 or -1
            if(sentiment == '+1'):
                f.write("+1\n")
            else:
                f.write("-1\n")

if __name__ == '__main__':
    main()
