from PIL import Image
import numpy as np
import sys
import glob
import json
import os

# python find_local_cues.py <dataset directory> <mask directory>
"""
dataset directory must include:
- pedestrian_dataset_folds/ with directories fold1/ to fold5/, and fold_dict.json
mask directory must include:
- video_0001/, video_0002/, etc for each video in which a pedestrian is present in JAAD
"""
dataDir: str = sys.argv[1]
maskDir: str = sys.argv[2]

# Resize pedestrian bounding box to height * 2 (this is the region in which we look for cues)
def expandBbox(topLeft, bottomRight):
	width = bottomRight[0] - topLeft[0]
	height = bottomRight[1] - topLeft[1]
	centerX = topLeft[0] + width // 2
	centerY = topLeft[1] + height // 2
	
	# Don't exceed image bounds when expanding
	newTopLeftX = max(centerX - height, 0)
	newTopLeftY = max(centerY - height, 0)
	newBottomRightX = min(centerX + height, 1920)
	newBottomRightY = min(centerY + height, 1080)
	return (newTopLeftX, newTopLeftY, newBottomRightX, newBottomRightY)

def featuresInBbox(bbox, mask):
	(topLeftX, topLeftY, bottomRightX, bottomRightY) = bbox
	patch = mask[topLeftX:bottomRightX, topLeftY:bottomRightY]
	labels, counts = np.unique(patch, return_counts=True)
	
	threshold = patch.size * 0.05
	labels = labels[counts >= threshold]
	
	return {
		"road": False,
		"sidewalk": False,
		"traffic sign": 50 in labels,
		"vehicle": 100 in labels,
		"traffic light": 150 in labels
	}

fold_dict_filename = '/pedestrian_dataset_folds/fold_dict.json'
with open(dataDir + fold_dict_filename, 'r') as f:
	fold_dict = json.load(f)
for json_filename in fold_dict:
	json_path = dataDir + "/" + fold_dict[json_filename] + "/" + json_filename
	with open(json_path, 'r') as f:
		ped_json = json.load(f)

	videoDir = maskDir + "/" + ped_json['video']
	frames = ped_json["frame_data"]
	for i, frame in enumerate(frames):
		index = frame["frame_index"]
		bbox = expandBbox(frame["bb_top_left"], frame["bb_bottom_right"])
		mask = np.array(Image.open(videoDir + "/" + str(index) + ".png"))
		features = featuresInBbox(bbox, mask)
		# add features to JSON
		ped_json["frame_data"][i].update(features)
	
	with open(videoDir + "/" + json_filename, "w") as f:
		json.dump(ped_json, f)
