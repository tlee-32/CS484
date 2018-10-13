import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def plotElbow():
    return

def plotClusters():
    return

def plotSparsePCAExplainedVariance(data, components):
    pca = TruncatedSVD(n_components=components).fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
