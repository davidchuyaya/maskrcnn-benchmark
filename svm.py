from sklearn.svm import LinearSVC
from joblib import dump, load
import numpy as np
import sys

train = True
# Must contain "/pedestrian_dataset_folds"
dataDir = sys.argv[1]

filename = "svm.joblib"

# Returns (data, labels), each a 5 element array of (fold_data, fold_labels)
def getFeatures(directory: str):
	with open(directory + "/pedestrian_dataset_folds/fold_dict.json", "r") as f:
		fold_dict = json.load(f)
	count = 0
	data = []
	labels = []
	
	foldRoot = directory + "/pedestrian_dataset_folds"
	for fold in os.listdir(foldRoot):
		if fold == "fold_dict.json":
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
				frame_features.extend(frame_dict['resnet_feature'][0])
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
				
				fold_data.append(np.array(frame_features))	
				fold_labels.append(int(ped_json['crossing']))
			count += 1
			
		data.append(fold_data)
		labels.append(fold_labels)
	return data, labels

data, labels = getFeatures(dataDir)

# TODO change to leave-one-out 5-fold validation
xTr = data[:-1]
xTr = xTr.reshape(-1, xTr.shape[-1])
yTr = labels[:-1]
yTr = yTr.reshape(-1, yTr.shape[-1])
xTe = data[-1]
yTe = labels[-1]
	
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
# print("Predictions: " + str(predictions))
misclassifications = np.sum(np.rint(predictions) != yTe)
error = misclassifications / len(yTe)
print("Error: " + str(error))