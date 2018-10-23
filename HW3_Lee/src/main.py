# Library imports
import time
import smart_open
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# Custom file imports
from classifier.helper.plot import *
from classifier.kmeans import KMeansClassifier
from preprocess.filetokenizer import readRows, tokenizeFloats
from preprocess.vectorizer import createCorpus, createTFIDF
from preprocess.sparsifier import sparsifyTermDocument

def main():
    startTime = time.time()
    """
    # Iris data classification
    irisData = np.asarray(tokenizeFloats("./data/test_iris.data"))
    irisData = normalize(irisData)
    kMeansIris = KMeansClassifier(clusters=3, metric='euclidean')
    labels = kMeansIris.fit(irisData)
    plotSilhouetteLine(kMeansIris, irisData)
    plotSilhouetteGraph(kMeansIris, irisData, dataSize=len(irisData), clusters=3, metric='euclidean')
    """
    # Retrieve raw data
    featureData = readRows("./data/features.data", loadFile=True, isFeatureFile=True)
    inputData = readRows("./data/input.data", loadFile=True)
    
    # Re-create the corpus to its original text form
    corpus = createCorpus(inputData, featureData)
    
    # Main feature reduction using TF-IDF
    print('Reducing features...')
    tfidf = createTFIDF()
    data = tfidf.fit_transform(corpus)
    
    """
    # Alternative feature reduction using Doc2Vec
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(documents, dm=0, dbow_words=0, hs=0, negative=3, sample=1e-4, workers=3)
    data = model.docvecs.doctag_syn0
    """

    """
    # Plot internal metrics
    #kMeansText = KMeansClassifier(clusters=7, metric='cosine')
    #plotSilhouetteGraph(kMeansText, sparseInput, dataSize=sparseInput.shape[0], clusters=7, metric='cosine')
    #plotSilhouetteLine(kMeansText, sparseInput)
    """

    # Perform classification
    print('Classifying...\n')
    classifyClusters(data, "textPredictions.data")
    print('\Clusters successfully written to textPredictions2.data (%d seconds)' % (time.time() - startTime))

"""
    Writes each predicted cluster to the output file
"""
def classifyClusters(data, fileName):
    labels = None
    maxClusterSize = 0
    optimalMaxClusterSize = 2800 # derived through multiple observations and runs of the algorithm
    
    # Choose the labels with the optimal max cluster size
    while maxClusterSize < optimalMaxClusterSize:
      kMeansText = KMeansClassifier(clusters=7, metric='cosine')
      labels = kMeansText.fit(data)
      score = silhouette_score(data, labels, metric='cosine')
      # Find size of each cluster
      clusterSize = [len(labels[labels==x]) for x in range(7)]
      # Max cluster for the current result
      maxClusterSize = max(clusterSize)
      print('Labels:', clusterSize)
      print('Max Cluster Size:', maxClusterSize)
      print('Silhouette Coefficient:', score, '\n')
    
    with smart_open.smart_open("./data/"+fileName, "w") as f:
        labels += 1 # change to 1-based numbering
        for label in labels:
            l = "%d\n" % label
            f.write(l)

if __name__ == '__main__':
    main()
