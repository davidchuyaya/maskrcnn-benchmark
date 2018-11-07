from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from PIL import Image
import numpy as np
import sys
import glob
import json

# call with <directory> <mode>
# directory. Must only contain image files.
directory: str = sys.argv[1]
# mode = "cpu" or "cuda"
mode: str = sys.argv[2]


# update the config options with the config file
cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", mode])
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


def get_file_names() -> list:
    types = ["*.jpg", "*.png"]
    names = []
    for extension in types:
        names += glob.glob1(directory, extension)
    return names

def load(filename: str):
    pil_image = Image.open(directory + "/" + filename).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def predict(img):
    predictions = coco_demo.compute_prediction(img)
    return coco_demo.select_top_predictions(predictions)

def save(predictions, filename: str):
    labels = predictions.get_field("labels").tolist()
    bounding_boxes = predictions.bbox.tolist()
    json_objs = []

    for i in range(len(labels)):
        label = COCODemo.CATEGORIES[labels[i]]
        bounding_box = bounding_boxes[i]
        json_objs.append({"label": label, "bbox": bounding_box})

    json_name = filename.split(".")[0]
    f = open(directory + "/" + json_name + ".json", "w")
    json.dump(json_objs, f)
    f.close()

# load image and then run prediction
for f in get_file_names():
    image = load(f)
    predictions = predict(image)
    save(predictions, f)
