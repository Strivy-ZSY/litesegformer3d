#!/bin/sh

DATASET_PATH=../DATASET_Synapse
CHECKPOINT_PATH=unetr_pp/evaluation/unetr_pp_synapse_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

PYTHONUNBUFFERED=1 nohup python unetr_pp/run/run_training.py 3d_fullres segformer3d_trainer_synapse 2 0 -val > eva_segmebest_synapse_2.6.txt 2>&1 & 
