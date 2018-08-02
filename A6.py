import pandas as pnd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




path = '/Users/vishwasbordekar/dal/acads/dataWare/A6/a6_temp.csv'
df = pnd.read_csv(path,engine='python',names=["corpus","lang"])
vectorizer = CountVectorizer()
vectorizer
X = vectorizer.fit_transform(df["corpus"])
X
print(X)

tsvd = TruncatedSVD(n_components=20)
X_tsvd = tsvd.fit(X).transform(X)
print(X_tsvd)

#plt.scatter(X_tsvd[:, 0], X_tsvd[:, 1])
#plt.show()
pca= PCA(n_components=2)
features = pca.fit(X_tsvd)
print(pca.singular_values_)
print(pca.explained_variance_ratio_)
print("===================")
print(pca.components_)

#code reffered from:- https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X_tsvd[:, 0], X_tsvd[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');