#!/bin/sh

DATASET_PATH=../DATASET_Synapse

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_synapse
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

# train
python unetr_pp/run/run_training.py 3d_fullres litesegformer3d_trainer_synapse 2 0

# inference
# python unetr_pp/run/run_training.py 3d_fullres litesegformer3d_trainer_synapse 2 0 --valbest --validation_only
