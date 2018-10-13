import os
import smart_open
import pickle
from nltk.tokenize import word_tokenize

"""
    Reads and tokenizes each row in the test file.

    return - tokenized rows
"""
def tokenizeInput(fileName):
    tokens = []
    # read training file
    for row in smart_open.smart_open(fileName, encoding="utf-8"):
        tokenizedRow = word_tokenize(row)
        tokenizedRow = [float(item) for item in tokenizedRow] # convert each row to an integer
        tokens.append(tokenizedRow)
    return tokens

"""
    Reads and tokenizes the term-frequency.

    return - tokenized term-frequencies
"""
def tokenizeTermFrequency(fileName):
    tokens = []
    # read training file
    for row in smart_open.smart_open(fileName, encoding="utf-8"):
        row = word_tokenize(row)
        tokenizedRow = []
        # create tokenized tuples from termId:termCount
        for idx in range(0, len(row), 2):
            termId = row[idx]
            termCount = row[idx+1]
            tokenizedRow.append((int(termId), int(termCount)))
        tokens.append(tokenizedRow)
    return tokens

"""
    Read rows from raw test file OR load a pickled file to
    deserialize the object. Pickled files assume that the test file
    has already been read and checkpointed. Raw test files will be
    tokenized and pickled.

    return - tokenized rows a generator
"""
def readRows(fileName, loadFile=False):
    tokenFile = renameFileExtension(fileName, 'data', 'tokens')
    tokens = []
    if(loadFile):
        # deserialize objects if pickled file already exists
        with smart_open.smart_open(tokenFile, "rb") as f:
            tokens = pickle.load(f, encoding="utf-8")
    else:
        # serialize and pickle the objects to files with pickled extension
        tokens = tokenizeTermFrequency(fileName) 
        serializeObject(tokenFile, tokens)
        
    return tokens

"""
    Serializes the object into a file.
"""
def serializeObject(fileName, obj):
    with smart_open.smart_open(fileName, "wb") as f:
        pickle.dump(obj, f)

def renameFileExtension(fileName, oldExt, newExt):
    fileExtensionIdx = fileName.rfind(oldExt)
    newFileName = fileName[:fileExtensionIdx] + newExt
    return newFileName