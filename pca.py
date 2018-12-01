from sklearn.decomposition import PCA
from joblib import dump, load
import numpy as np

# TODO: Take in X, train
X = []
train = True

filename = "pca.joblib"

if train:
	# set random_state = 0 for replicable results
	pca = PCA(n_components=100, random_state=0)
	pca.fit(X)
	print("Singular values: " + str(pca.singular_values_))
	dump(pca, filename)
else:
	pca = load(filename)

transformed = pca.transform(X)
print("Transformed X: " + str(transformed))