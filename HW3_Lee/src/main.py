# Library imports
import time
import smart_open
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
# Custom file imports
from classifier.helper.plot import *
from preprocess.filetokenizer import readRows, tokenizeInput
from preprocess.sparsifier import sparsifyTermDocument
from classifier.kmeans import KMeansClassifier

def main():
    startTime = time.time()
    # Retrieve data
    irisData = np.asarray(tokenizeInput("./data/test_iris.data"))
    inputData = readRows("./data/input.data", loadFile=True)
    
    # Convert data into csr matrix
    sparseInput = sparsifyTermDocument(inputData)
    #reduced_data = TruncatedSVD(n_components=2000).fit_transform(reduced_data)
    # Iris data
    #kMeansIris = KMeansClassifier(clusters=3)
    #classifyClusters(kMeansIris, irisData, "irisPredictions.data")
    
    #plotSparsePCAExplainedVariance(sparseInput, components=2000)
    
    kMeansText = KMeansClassifier(clusters=7, metric='cosine')
    classifyClusters(kMeansText, sparseInput, "textPredictions2.data")
    print('\nMolecle activity successfully written to predictions.data (%d seconds)' % (time.time() - startTime))

"""
    Writes each predicted cluster to the output file
"""
def classifyClusters(classifier, dataMatrix, fileName):
    with smart_open.smart_open("./data/"+fileName, "w") as f:
        labels = classifier.fit(dataMatrix)
        labels += 1 # change to 1-based numbering
        for label in labels:
            l = "%d\n" % label
            f.write(l)

if __name__ == '__main__':
    main()
