from sklearn.svm import LinearSVC
from joblib import dump, load
import numpy as np

# TODO: Take in xTr, yTr, xTe, yTe, train
# Assumes y = 0 or 1
xTr = []
yTr = []
xTe = []
yTe = []
train = True

filename = "svm.joblib"

if train:
	# set random_state = 0 for replicable results
	classifier = LinearSVC(penalty="l1", dual=False, random_state=0)
	classifier.fit(xTr, yTr)
	print("Weights: " + str(classifier.coef_))
	dump(classifier.coef_, filename)
else:
	classifier = load(filename)

# evaluate
predictions = classifier.predict(xTe)
print("Predictions: " + str(predictions))
misclassifications = np.sum(np.rint(predictions) != yTe)
error = misclassifications / len(yTe)
print("Error: " + str(error))