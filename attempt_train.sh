#!/bin/bash
sbatch --requeue --gres=gpu:2 --mem=64g --nodelist=hinton train.sh
