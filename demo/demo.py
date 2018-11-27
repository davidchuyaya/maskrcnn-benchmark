from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import requests
from io import BytesIO
from PIL import Image
import numpy as np

config_file = "../configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

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
    img.save("demo.png")

# load image and then run prediction
image = load("http://farm1.staticflickr.com/117/307382050_53dc21e48a_z.jpg")
predictions = coco_demo.run_on_opencv_image(image)
save(predictions)
