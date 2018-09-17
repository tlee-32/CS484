import smart_open
from preprocess.filetokenizer import readReviews
from knn.crossvalidation import *
from knn.knn import *

def main():
    reviews = list(readReviews("./data/train/train.data", loadFile=True, isTrainingFile=True))[0]
    ########Cross-validation########
    #k = findOptimalKForKNN(reviews)
    #return
    ################################
    testReviews = list(readReviews("./data/test/test.data", loadFile=True, isTrainingFile=False))[0]
    knn = KNNClassifier(k=7)
    knn.fit(reviews, retrain=True) # build Doc2Vec model with training data
    sentiment = classifySentimentWithKNN(knn, testReviews)
    print('Sentiments successfully written to predictions.data')

"""
    Classifies the sentiment for each of the 18560 Amazon reviews
    of a baby product and writes it to an output file.
"""
def classifySentimentWithKNN(knn, testData):
    with smart_open.smart_open("./data/predictions/predicitions.data", "w") as f:
        for review in testData:
            sentiment = knn.classify(review) # sentiment as +1 or -1
            if(sentiment == '+1'):
                f.write("+1\n")
            else:
                f.write("-1\n")

if __name__ == '__main__':
    main()
