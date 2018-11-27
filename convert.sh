#!/bin/bash
python convert_cityscapes_to_coco.py /scratch/datasets/cityscapes /scratch/datasets/cityscapes

#rename directories
cd /scratch/datasets/cityscapes
mv gtFine old_gtFine
mv leftImg8bit gtFine
