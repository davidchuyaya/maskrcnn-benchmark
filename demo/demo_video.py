from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2 as cv
import numpy as np

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


# load video and then run prediction
cam = cv.VideoCapture("/scratch/datasets/JAAD_clips/video_0001.mp4")
codec = cv.VideoWriter_fourcc(*"MJPG")
out = cv.VideoWriter("output.avi", codec, 30.0, (int(cam.get(3)), int(cam.get(4))))
i = 0
while True:
    ret, img = cam.read()
    if not ret:
        break
    predictions = coco_demo.run_on_opencv_image(img)
#    cv.imwrite("./video/" + str(i) + ".png", predictions)
    i += 1
    if i == 5:
        break    
    out.write(predictions)
cam.release()
out.release()
