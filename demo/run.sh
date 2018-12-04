#!/bin/bash
#sbatch --requeue --nodelist=tripods-compute02,hinton find_local_cues.sh

sbatch --requeue --gres=gpu:1 --mem=32g --nodelist=tripods-compute02,hinton create_segmentations.sh
