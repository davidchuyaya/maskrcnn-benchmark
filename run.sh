#!/bin/bash
cd ~
./download_cityscapes.sh

cd ~/maskrcnn_benchmark
python convert_cityscapes_to_coco.py /scratch/datasets/cityscapes /scratch/datasets/cityscapes

#rename directories
cd /scratch/datasets/cityscapes
mv gtFine old_gtFine
mv leftImg8bit gtFine

./attempt_train.sh