# Util file for cleaning and filtering raw text.
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize

"""
    Removes punctuation characters from a given string

    return - removed punctuations for string
"""
def removePunctuation(rawString):
    excludePunctuation = str.maketrans('', '', punctuation) # translation table that maps punctuation chars to None
    return rawString.translate(excludePunctuation)

"""
    Normalizes the token's case, whitespaces, and punctuation.

    return - normalized string
"""
def normalizeString(rawString):
    norm = rawString.lower().strip() # convert to lowercase and strip whitespaces
    norm = removePunctuation(norm)
    return norm

"""
    Removes uninformative English words such as
    "the", "i", "as", "a", etc in a list of tokens

    return - removed stop words for string
"""
def removeStopWords(tokens):
    stopWords = set(stopwords.words('english'))
    filtered = [word for word in tokens if not word in stopWords]
    return filtered

"""
    Filters and tokenizes a document

    return - filtered and tokenized document
"""
def tokenizeDocument(doc):
    normDoc = normalizeString(doc) # remove case, whitespaces, and punctuation
    tokenizedDoc = removeStopWords(word_tokenize(normDoc)) # remove uninformative words from tokenized list
    return tokenizedDoc
