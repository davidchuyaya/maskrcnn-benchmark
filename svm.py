from sklearn.svm import LinearSVC
from joblib import dump, load
import numpy as np
import json
import sys
import os

train = True
# Must contain "/pedestrian_dataset_folds"
dataDir = sys.argv[1]

filename = "svm.joblib"

# Returns (data, labels), each a 5 element array of (fold_data, fold_labels)
def getXY(directory: str):
	count = 0
	num_signs = 0
	num_cars = 0
	num_lights = 0
	data = []
	labels = []
	
	foldRoot = directory + "/pedestrian_dataset_folds"
	for fold in os.listdir(foldRoot):
		if not os.path.isdir(foldRoot + "/" + fold):
			continue
		
		fold_data = []
		fold_labels = []
		
		for ped in os.listdir(foldRoot + "/" + fold):
			with open(foldRoot + "/" + fold + "/" + ped, "r") as f:
				ped_json = json.load(f)

			ped_id = ped_json['video'] + '-' + ped_json['name']
			frame_data = ped_json['frame_data']
			for frame_dict in frame_data:
				frame_features = []

				resnet = np.array(frame_dict['resnet_feature'][0])
				resnet = (resnet - np.mean(resnet)) / np.std(resnet)
				frame_features.extend(resnet)
				frame_features.extend(frame_dict['attrib_vector'][0])
				frame_features.extend(frame_dict['global_cue_vector'])
				frame_features.append(int(frame_dict['standing']))
				frame_features.append(int(frame_dict['looking']))
				
				# add custom features
				# frame_features.append(int(frame_dict["road"]))
				# frame_features.append(int(frame_dict["sidewalk"]))
				frame_features.append(int(frame_dict["traffic sign"]))
				frame_features.append(int(frame_dict["vehicle"]))
				frame_features.append(int(frame_dict["traffic light"]))
				num_signs += int(frame_dict["traffic sign"])
				num_cars += int(frame_dict["vehicle"])
				num_lights += int(frame_dict["traffic light"])
				
				fold_data.append(np.array(frame_features))	
				fold_labels.append(int(ped_json['crossing']))
			count += 1
			
		data.append(fold_data)
		labels.append(fold_labels)
	print("Num signs: " + str(num_signs))
	print("Num cars: " + str(num_cars))
	print("Num lights: " + str(num_lights))
	return data, labels
	
# num ranges from 0 to 4
def leaveOneOut(foldToLeaveOut: int, X, Y):
	xTr = []
	yTr = []
	xTe = []
	yTe = []
	
	for i in range(len(X)):
		foldData = X[i]
		foldLabels = Y[i]
		
		if i == foldToLeaveOut:
			xTe = foldData
			yTe = foldLabels
		else:
			xTr += foldData
			yTr += foldLabels
			
	return xTr, yTr, xTe, yTe
	
def errorAfterTrainingOn(xTr, yTr, xTe, yTe):
	# set random_state = 0 for replicable results
	classifier = LinearSVC(penalty="l1", dual=False, random_state=0)
	classifier.fit(xTr, yTr)
	print("Weights: " + str(classifier.coef_))

	# evaluate
	predictions = classifier.predict(xTe)
	misclassifications = np.sum(np.rint(predictions) != yTe)
	error = misclassifications / len(yTe)
	print("Error: " + str(error))
	return error


data, labels = getXY(dataDir)
error = np.empty(5)
numFolds = 5
for i in range(numFolds):
	xTr, yTr, xTe, yTe = leaveOneOut(i, data, labels)
	error[i] = errorAfterTrainingOn(xTr, yTr, xTe, yTe)
print("Std dev: " + str(np.std(error)))
print("Mean error: " + str(np.mean(error)))
