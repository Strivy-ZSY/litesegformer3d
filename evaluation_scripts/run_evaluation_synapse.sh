#!/bin/sh

DATASET_PATH=../DATASET_Synapse
CHECKPOINT_PATH=lsf3d/evaluation/lsf3d_synapse_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export lsf3d_preprocessed="$DATASET_PATH"/lsf3d_raw/lsf3d_raw_data/Task02_Synapse
export lsf3d_raw_data_base="$DATASET_PATH"/lsf3d_raw

PYTHONUNBUFFERED=1 nohup python lsf3d/run/run_training.py 3d_fullres segformer3d_trainer_synapse 2 0 -val > eva_segmebest_synapse_2.6.txt 2>&1 & 
