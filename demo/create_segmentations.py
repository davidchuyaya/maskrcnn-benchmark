from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
import cv2 as cv
import sys
import json

# python create_segmentations.py <dataset directory> <output directory>
# NOTE: Must run on GPU with CUDA
"""
dataset directory must include directories:
- JAAD_clips/ with videos
- pedestrian_dataset_folds/ with directories fold1/ to fold5/, and fold_dict.json
"""
dataDir: str = sys.argv[1]
outDir: str = sys.argv[2]

def loadCOCOPredictor():
	cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml")
	return COCODemo(cfg)

def loadTrafficSignPredictor():
	cfg.merge_from_file("../configs/traffic_sign.yaml")
	return COCODemo(cfg)
	
# Returns map from videoName to a list of frames
def getFramesDict(directory: str):
	fold_dict_filename = '/pedestrian_dataset_folds/fold_dict.json'
	with open(directory + fold_dict_filename, 'r') as f:
		fold_dict = json.load(f)
	num_frames = 30
	frames_to_process = set()
	for json_filename in fold_dict:
		json_path = directory + "/" + fold_dict[json_filename] + "/" + json_filename
		with open(json_path, 'r') as f:
			ped_json = json.load(f)
		video_name = ped_json['video']
		first_frame = ped_json['frame_data'][0]
		start = first_frame['frame_index']
		for idx in range(start, start + num_frames):
			frames_to_process.add(video_name + '-' + str(idx))
	frames_dict = {}
	for frame in frames_to_process:
		video, idx = frame.split('-')
		if video not in frames_dict:
			frames_dict[video] = []
		frames_dict[video].append(int(idx))
	for video in frames_dict:
		frames_dict[video] = sorted(frames_dict[video])
	return frames_dict
	
def getVideoFrames(videoName: str, frames: list):
	video = cv.VideoCapture(videoName)
	i = 1
	while True:
		ret, img = video.read()
		if not ret:
			video.release()
			return
		if i in frames:
			yield (i, img)
		i += 1
		
def predictLabelsAndSegmentations(img, predictor):
	predictions = predictor.compute_prediction(img)
	predictions = predictor.select_top_predictions(predictions)
	labels = predictions.get_field("labels").numpy()
	masks = predictions.get_field("mask").numpy()
	segmentations = []
	
	for mask in masks:
		thresh = mask[0, :, :, None]
		_, contours, hierarchy = cv.findContours(
			thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
		)
		segmentations.append(contours)
		
	segmentations = np.array(segmentations).flatten()
	return (labels, segmentations)
		
def predictFrame(img, cocoPredictor, trafficSignPredictor):
	(cocoLabels, cocoSegmentations) = predictLabelsAndSegmentations(img, cocoPredictor)
	(_, trafficSignSegmentations) = predictLabelsAndSegmentations(img, trafficSignPredictor)
	
	carIndices = (cocoLabels >= 3) & (cocoLabels <= 9)
	trafficLightIndices = cocoLabels == 10
	
	segmentations = []
	for vehicle in cocoSegmentations[carIndices]:
		segmentations.append({"label": "car", "segmentation": vehicle})
	for trafficLight in cocoSegmentations[trafficLightIndices]:
		segmentations.append({"label": "traffic light", "segmentation": trafficLight})
	for trafficSign in trafficSignSegmentations:
		segmentations.append({"label": "traffic sign", "segmentation": trafficSign})
	return segmentations
	
class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		if isinstance(o, numpy.int32):
			return int(o)  
		return json.JSONEncoder.default(self, obj)

cocoPredictor = loadCOCOPredictor()
trafficSignPredictor = loadTrafficSignPredictor()

frames_dict = getFramesDict(dataDir)
for videoName, frames in frames_dict.items():
	frame_data = []
	print("Video: " + videoName + ", frames: " + str(frames))
	
	for frame, img in getVideoFrames(dataDir + "/JAAD_clips/" + videoName + ".mp4", frames):
		predictions = predictFrame(img, cocoPredictor, trafficSignPredictor)
		frame_data.append({"frame_index": frame, "segmentations": predictions})
		print("Appended for " + videoName + "-" + str(frame))
		
	with open(outDir + "/" + videoName + ".json", "w") as f:
		print("Saving segmentation for " + videoName)
		json.dump(frame_data, f, cls=NumpyEncoder)
