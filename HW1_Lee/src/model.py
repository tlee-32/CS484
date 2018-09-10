# Holds functions to vectorize documents using gensim

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

def trainDoc2VecModel(trainDocs, vectorSize, window, minCount, epochs):
    model = Doc2Vec(trainDocs, vector_size=vectorSize, window=window, min_count=minCount, workers=4, epochs=epochs)
    model.train(trainDocs, total_examples=model.corpus_count, epochs=model.epochs)
    return model
