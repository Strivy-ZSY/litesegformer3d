#!/bin/sh

DATASET_PATH=../DATASET_Acdc
CHECKPOINT_PATH=lsf3d/evaluation/lsf3d_acdc_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export lsf3d_preprocessed="$DATASET_PATH"/lsf3d_raw/lsf3d_raw_data/Task01_ACDC
export lsf3d_raw_data_base="$DATASET_PATH"/lsf3d_raw

python lsf3d/run/run_training.py 3d_fullres segformer3d_trainer_acdc 1 0 --valbest --validation_only
