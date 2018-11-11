#!/usr/bin/python
#
# Convert instances from png files to a dictionary
#

from __future__ import print_function, absolute_import, division
import os, sys

import numpy as np

# Cityscapes imports
from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *
import cv2


def instances2dict_with_polygons(imageFileList, verbose=False):
    # print(imageFileList)
    imgCount = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue

            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            # instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)

                # print(cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE))

                _, contour, hier = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict["contours"] = polygons

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=" ")
            sys.stdout.flush()

    if verbose:
        print("")

    # print(instanceDict)
    return instanceDict


def main(argv):
    fileList = []
    if len(argv) > 2:
        for arg in argv:
            if "png" in arg:
                fileList.append(arg)
    instances2dict_with_polygons(fileList, True)


if __name__ == "__main__":
    main(sys.argv[1:])
