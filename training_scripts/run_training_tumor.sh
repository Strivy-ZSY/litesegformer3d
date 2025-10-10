#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_tumor
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task03_tumor
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

# PYTHONUNBUFFERED=1 python unetr_pp/run/run_training.py 3d_fullres segformer3d_trainer_tumor 3 0 --valbest --validation_only


PYTHONUNBUFFERED=1 nohup python unetr_pp/run/run_training.py 3d_fullres segformer3d_trainer_synapse 2 0  > inf_tum_liteseg.txt --valbest --validation_only 2>&1 &
