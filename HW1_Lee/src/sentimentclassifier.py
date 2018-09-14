import smart_open
from preprocess.filetokenizer import readReviews
from knn.knn import KNNClassifier

def main():
    reviews = list(readReviews("../data/train/train.data", loadFile=True, isTrainingFile=True))[0]
    testReviews = list(readReviews("../data/test/test.data", loadFile=True, isTrainingFile=False))[0]
    knn = KNNClassifier(k=4)
    knn.fit(reviews, retrain=False) # build Doc2Vec model with training data
    sentiment = classifySentimentWithKNN(knn, testReviews)
    print('Sentiments successfully written to output.data')

"""
    Classifies the sentiment for each of the 18560 Amazon reviews
    of a baby product and writes it to an output file.
"""
def classifySentimentWithKNN(knn, testData):
    with smart_open.smart_open("../data/predictions/predicitions.data", "w") as f:
        for review in testData:
            sentiment = knn.classify(review) # sentiment as +1 or -1
            if(sentiment == '+1'):
                f.write("+1\n")
            else:
                f.write("-1\n")

if __name__ == '__main__':
    main()
