#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 					http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys
import numpy as np


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError("Argument xyxy must be a list, tuple, or numpy array.")


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys


def poly_to_segmentation(poly):
    # poly = array of points. A point = array of ints, [[x1, y1], [x2, y2], ...].
    # segmentation = array of array of points, [x1, y1, x2, y2, ...].
    return np.array(poly).reshape([1, -1])


def area_of_poly(poly):
    # poly = array of points. A point = array of ints, [[x1, y1], [x2, y2], ...].
    sum = 0
    for i in range(len(poly)):
        x0 = poly[i][0]
        y0 = poly[i][1]
        # j = next point, wrapping around to 0
        j = i + 1 if i < len(poly) - 1 else 0
        x1 = poly[j][0]
        y1 = poly[j][1]
        sum += x0 * y1 - x1 * y0
    return abs(sum / 2)


def convert_cityscapes_instance_only(data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = ["gtFine_val", "gtFine_train", "gtFine_test"]
    ann_dirs = ["gtFine/val", "gtFine/train", "gtFine/test"]
    json_name = "instancesonly_filtered_%s.json"
    postfix = "gtFine_polygons.json"
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    categories_to_convert = ["traffic sign"]

    for data_set, ann_dir in zip(sets, ann_dirs):
        print("Starting %s" % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        for root, _, files in os.walk(ann_dir):
            for filename in files:
                if not filename.endswith(postfix):
                    continue

                if len(images) % 50 == 0:
                    print(
                        "Processed %s images, %s annotations"
                        % (len(images), len(annotations))
                    )
                json_ann = json.load(open(os.path.join(root, filename)))
                image = {}
                image["id"] = img_id
                img_id += 1

                image["width"] = json_ann["imgWidth"]
                image["height"] = json_ann["imgHeight"]
                image["file_name"] = filename[: -len(postfix)] + "leftImg8bit.png"
                images.append(image)

                objects = json_ann["objects"]

                for obj in objects:
                    object_cls = obj["label"]
                    if object_cls not in categories_to_convert:
                        continue  # skip categories

                    poly = obj["polygon"]
                    segmentation = poly_to_segmentation(poly)

                    ann = {}
                    ann["id"] = ann_id
                    ann_id += 1
                    ann["image_id"] = image["id"]
                    ann["segmentation"] = segmentation.tolist()

                    if object_cls not in category_dict:
                        category_dict[object_cls] = cat_id
                        cat_id += 1
                    ann["category_id"] = category_dict[object_cls]
                    ann["iscrowd"] = 0
                    ann["area"] = area_of_poly(poly)
                    ann["bbox"] = xyxy_to_xywh(polys_to_boxes([segmentation])).tolist()[
                        0
                    ]

                    annotations.append(ann)

        ann_dict["images"] = images
        categories = [
            {"id": category_dict[name], "name": name} for name in category_dict
        ]
        ann_dict["categories"] = categories
        ann_dict["annotations"] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), "w") as outfile:
            json.dump(ann_dict, outfile)


if __name__ == "__main__":
    datadir = sys.argv[1]
    outdir = sys.argv[2]
    convert_cityscapes_instance_only(datadir, outdir)
