#!/bin/sh

DATASET_PATH=../DATASET_Synapse
CHECKPOINT_PATH=lsf3d/evaluation/litesegformer3d_synapse_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export litesegformer3d_preprocessed="$DATASET_PATH"/litesegformer3d_raw/litesegformer3d_raw_data/Task02_Synapse
export litesegformer3d_raw_data_base="$DATASET_PATH"/litesegformer3d_raw

python lsf3d/run/run_training.py 3d_fullres litesegformer3d_trainer_synapse 2 0 --valbest --validation_only
