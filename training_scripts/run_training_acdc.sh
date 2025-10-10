#!/bin/sh

DATASET_PATH=../DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_acdc
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

# python unetr_pp/run/run_training.py 3d_fullres segformer3d_trainer_acdc 1 0

PYTHONUNBUFFERED=1 nohup python unetr_pp/run/run_training.py 3d_fullres segformer3d_trainer_acdc 1 0 > acdc_org_lw.txt 2>&1 &