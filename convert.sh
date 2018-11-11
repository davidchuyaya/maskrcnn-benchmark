#!/bin/bash
rm -rf out
mkdir out
python convert_cityscapes_to_coco.py /scratch/datasets/cityscapes out
