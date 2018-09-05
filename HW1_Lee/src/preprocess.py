from nltk.tokenize import word_tokenize
from string import punctuation


"""
    Helper function to remove punctuation characters, uppercases,
    and tabs for a string.
"""
def getFilteredString(s):
    excludePunctuation = str.maketrans('', '', punctuation) # translation table that maps punctuation chars to None
    return s.lower().strip().translate(excludePunctuation)

"""
    Read each review (lowercase and w.o. punctuation) in the file
    and categorize its sentiment based on +1 or -1.

    return - dictionary
"""
def readAndCategorizeReviews(fileName):
    categorizedReviews = {'+1' : [], '-1' : []}
    cnt = 1
    with open(fileName, 'r') as fp:
        for review in fp:
            if cnt == 50: break
            sentiment = review[0:2] # grab +1 or -1
            review = review[2:] # continue reading the review
            categorizedReviews[sentiment].append(getFilteredString(review))
            cnt += 1
    return categorizedReviews

"""
    Takes in a 2D list and tokenizes the words from each item in the list.
"""
def getTokenizedWordsPerItem(reviews):
    words = []
    for review in reviews:
        words.append(word_tokenize(review))
    return words



reviews = readAndCategorizeReviews("../train.data")
words = getTokenizedWordsPerItem(reviews['-1'])
print(words)
