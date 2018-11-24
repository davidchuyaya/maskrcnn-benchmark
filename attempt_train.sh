#!/bin/bash
sbatch --requeue --gres=gpu:1 --mem=32g --nodelist=hinton train.sh
