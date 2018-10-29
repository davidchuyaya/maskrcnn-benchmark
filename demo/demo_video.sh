#!/bin/bash

~/anaconda3/etc/profile.d/conda.sh activate base

cd ~/maskrcnn-benchmark/demo
python demo_video.py
echo "hi"

