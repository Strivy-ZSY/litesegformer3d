#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_tumor
export litesegformer3d_preprocessed="$DATASET_PATH"/litesegformer3d_raw/litesegformer3d_raw_data/Task03_tumor
export litesegformer3d_raw_data_base="$DATASET_PATH"/litesegformer3d_raw

python lsf3d/run/run_training.py 3d_fullres litesegformer3d_trainer_tumor 3 0
