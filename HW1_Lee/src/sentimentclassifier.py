import datetime # TESTING
# Python 3
from filetokenizer import *
from crossvalidation import *
from model import *
import smart_open
import numpy as np

def main():
    start = datetime.datetime.now()

    reviews = list(readReviews("../train.data", pickleFileExists=True, isTrainingFile=True))[0]

    k = findOptimalKForKNN(reviews) # find optimal k using cross validation
    #knn = KNNClassifier(k)
    #knn.fit(reviews) # build Doc2Vec model of training data


    # classifySentimentWithKNN(knn, getTestData("../test.data"))
    end = datetime.datetime.now()
    totalSeconds = (end - start).total_seconds()
    print("Duration: {}".format(totalSeconds))

# DO NOT USE THIS UNTIL THE VERY END. PLEASE.
def classifySentimentWithKNN(knn, testData):
    with smart_open.smart_open("../output.data", "wb") as f:
        for review in testData:
            sentiment = knn.classify(review) # sentiment as +1 or -1
            f.write("%s\n" % sentiment)

def getTestData(fileName):
    yield readReviews(fileName)

if __name__ == '__main__':
    main()
