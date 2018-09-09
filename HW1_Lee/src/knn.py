import datetime # TESTING
# Python 3
from filetokenizer import *
from vector import *
import numpy

def main():
    start = datetime.datetime.now()
    reviews = readReviews("../train.data", pickleFileExists=True, isTrainingFile=True)
    trainDoc2VecModel(list(reviews)[0])
    end = datetime.datetime.now()
    totalSeconds = (end - start).total_seconds()
    print("Duration: {}".format(totalSeconds))



if __name__ == '__main__':
    main()
