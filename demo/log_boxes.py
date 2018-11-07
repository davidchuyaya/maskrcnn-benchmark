from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import sys

# mode = "cpu" or "cuda"
mode = sys.argv[1]

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", mode])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def save(arr):
    img = Image.fromarray(arr, "RGB")
    img.save("out/" + image_file)

# load image and then run prediction
image = load("https://user-images.githubusercontent.com/3080674/29361099-52eb370c-8286-11e7-8274-ceb4895fe0b9.png")
predictions = coco_demo.compute_prediction(image)
top_predictions = coco_demo.select_top_predictions(predictions)
out = str(top_predictions.get_field("labels"))
out += str(top_predictions.bbox)
f = open("out.json", "w")
f.write(out)
f.close()
