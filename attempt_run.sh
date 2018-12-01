#!/bin/bash
sbatch --requeue --gres=gpu:2 --mem=32g run.sh
