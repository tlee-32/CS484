from sklearn.feature_extraction.text import TfidfVectorizer

"""
  Create original corpus by mapping each (termId, termCount) to 
  its associated feature.
"""
def createCorpus(inputData, featureData):
    docs = []
    for item in inputData:
        doc = []
        for termId, termCount in item:
            # since feature file starts at line 1, accomodate for 0-based indexing
            word = featureData[termId - 1]
            doc.extend([word] * termCount)
        docs.append(doc)
    return docs

"""
    Configure a TFIDF Vectorizer that removes stop words
"""
def createTFIDF():
    vectorizer = TfidfVectorizer(stop_words='english',
                                 analyzer='word',
                                 max_df=0.9,
                                 norm='l2',
                                 tokenizer=tokenizer, 
                                 preprocessor=tokenizer,
                                 token_pattern=None)
    return vectorizer

def tokenizer(doc):
    return doc
