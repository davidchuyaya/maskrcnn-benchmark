#!/bin/bash
#cd ~
#./download_cityscapes.sh

#cd ~/maskrcnn-benchmark
#python convert_cityscapes_to_coco.py ~/cityscapes ~/cityscapes

#rename directories
#cd ~/cityscapes
#mv gtFine old_gtFine
#mv leftImg8bit gtFine

cp -r -n ~/cityscapes /scratch/datasets/cityscapes || true

cd ~/maskrcnn-benchmark
./train.sh
