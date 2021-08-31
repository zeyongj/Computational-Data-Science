import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def get_pca(X):
    """
    Transform data to 2D points for plotting. Should return an array with shape (n, 2).
    """
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html .
    # The following codes are also adapted from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html .
    from sklearn.decomposition import PCA
    flatten_model = make_pipeline(
        # TODO
        MinMaxScaler(),
        PCA(n_components = 2)
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2


def get_clusters(X):
    """
    Find clusters of the weather data.
    """
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html .
    from sklearn.cluster import KMeans
    model = make_pipeline(
        # TODO
        KMeans(n_clusters = 10, random_state = None) # As required in the instruction.
    )
    model.fit(X)
    # metric = model.inertia_
    # print ('Opposite of the value of X on the K-means objective is %g.' % metric)
    return model.predict(X)


def main():
    data = pd.read_csv(sys.argv[1])

    X = data.iloc[:, 1:] # TODO: First two colums are needed.
    y = data.iloc[:,0] # TODO: The Column of label (i.e. first column) is needed.
    
    X2 = get_pca(X)
    clusters = get_clusters(X)
    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=20)
    plt.savefig('clusters.png')

    df = pd.DataFrame({
        'cluster': clusters,
        'city': y,
    })
    counts = pd.crosstab(df['city'], df['cluster'])
    print(counts)


if __name__ == '__main__':
    main()
