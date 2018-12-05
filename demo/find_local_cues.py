from PIL import Image
import numpy as np
import sys
import json
import os

# python find_local_cues.py <dataset dir> <instance mask dir> <semantic mask dir> <output dir>
"""
dataset directory must include:
- pedestrian_dataset_folds/ with directories fold1/ to fold5/, and fold_dict.json
mask directories must include:
- video_0001/, video_0002/, etc for each video in which a pedestrian is present in JAAD
"""
dataDir: str = sys.argv[1]
instanceMaskDir: str = sys.argv[2]
semanticMaskDir: str = sys.argv[3]
outDir: str = sys.argv[4]

# Resize pedestrian bounding box to height * 2 (this is the region in which we look for cues)
def expandBbox(topLeft, bottomRight):
	width = bottomRight[0] - topLeft[0]
	height = bottomRight[1] - topLeft[1]
	centerX = topLeft[0] + width // 2
	centerY = topLeft[1] + height // 2
	
	# Don't exceed image bounds when expanding
	newTopLeftX = max(centerX - height * 2, 0)
	newTopLeftY = max(centerY - height * 2, 0)
	newBottomRightX = min(centerX + height * 2, 1920)
	newBottomRightY = min(centerY + height * 2, 1080)
	return (newTopLeftX, newTopLeftY, newBottomRightX, newBottomRightY)

def featuresInBbox(bbox, mask, mask_type, percentage=0.05):
	(topLeftX, topLeftY, bottomRightX, bottomRightY) = bbox
	
	if mask_type == 'semantic':
		if mask is None:
			return {
				"road": False,
				"sidewalk": False
			}
			
		topLeftX, topLeftY, bottomRightX, bottomRightY = bbox
		patch = mask[topLeftY : bottomRightY, topLeftX : bottomRightX]
		labels, counts = np.unique(patch, return_counts=True)
		
		threshold = int(patch.size * percentage)
		labels = labels[counts >= threshold]
		
		return {
			"road": 0 in labels,
			"sidewalk": 1 in labels,
		}	
	elif mask_type == 'instance':
		patch = mask[topLeftX:bottomRightX, topLeftY:bottomRightY]
		labels, counts = np.unique(patch, return_counts=True)
		
		threshold = int(patch.size * percentage)
		labels = labels[counts >= threshold]
		
		return {
			"traffic sign": 50 in labels,
			"vehicle": 100 in labels,
			"traffic light": 150 in labels
		}
	else:
		return {}

fold_dict_filename = '/pedestrian_dataset_folds/fold_dict.json'
with open(dataDir + fold_dict_filename, 'r') as f:
	fold_dict = json.load(f)
for json_filename in fold_dict:
	fold = fold_dict[json_filename]
	json_path = dataDir + "/" + fold + "/" + json_filename
	with open(json_path, 'r') as f:
		ped_json = json.load(f)

	instanceVideoDir = instanceMaskDir + "/" + ped_json['video']
	semanticVideoDir = semanticMaskDir + "/" + ped_json['video']
	frames = ped_json["frame_data"]
	for i, frame in enumerate(frames):
		index = frame["frame_index"]
		bbox = expandBbox(frame["bb_top_left"], frame["bb_bottom_right"])
		try:
			instanceMask = np.array(Image.open(instanceVideoDir + "/" + str(index) + ".png"))
			instanceFeatures = featuresInBbox(bbox, instanceMask, 'instance')
		except:
			print("Could not find " + instanceVideoDir + "/" + str(index) + ".png")
			instanceFeatures = {"traffic sign": False, "vehicle": False, "traffic light": False}
		try:
			semanticMask = np.array(Image.open(semanticVideoDir + "/" + str(index) + ".png"))
			semanticFeatures = featuresInBbox(bbox, semanticMask, 'semantic')
		except IOError:
			print("Could not find " + semanticVideoDir + "/" + str(index) + ".png")
			semanticFeatures = {"road": False, "sidewalk": False}
		# add features to JSON
		ped_json["frame_data"][i].update(instanceFeatures)
		ped_json["frame_data"][i].update(semanticFeatures)
	
	os.makedirs(outDir + "/" + fold, exist_ok=True)
	with open(outDir + "/" + fold + "/" + json_filename, "w") as f:
		json.dump(ped_json, f)
