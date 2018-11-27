#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml" TEST.IMS_PER_BATCH 2
echo "done boiz"
