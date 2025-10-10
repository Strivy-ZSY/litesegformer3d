#!/bin/sh

DATASET_PATH=../DATASET_Synapse

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_synapse
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

# PYTHONUNBUFFERED=1 nohup python unetr_pp/run/run_training.py 3d_fullres segformer3d_trainer_synapse 2 0  > synapse_segformer_8category.txt 2>&1 &

# over, test grid_minmax=2.6

PYTHONUNBUFFERED=1 nohup python unetr_pp/run/run_training.py 3d_fullres segformer3d_trainer_synapse 2 0  > inf_syn_liteseg.txt 2>&1 &