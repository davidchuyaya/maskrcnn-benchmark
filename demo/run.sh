#!/bin/bash
sbatch --requeue --mem=32g --nodelist=tripods-compute02 --gres=gpu:1 find_local_cues.sh
