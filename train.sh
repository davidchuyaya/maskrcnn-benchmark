#!/bin/bash
# python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1
python -m torch.distributed.launch --nproc_per_node=2 tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 SOLVER.MAX_ITER 360000 SOLVER.STEPS "(240000, 320000)" TEST.IMS_PER_BATCH 2
echo "done boiz"
