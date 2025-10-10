#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../lsf3d/evaluation/litesegformer3d_tumor_checkpoint
export litesegformer3d_preprocessed="$DATASET_PATH"/litesegformer3d_raw/litesegformer3d_raw_data/Task03_tumor
export litesegformer3d_raw_data_base="$DATASET_PATH"/litesegformer3d_raw


# Only for Tumor, it is recommended to train litesegformer3d first, and then use the provided checkpoint to evaluate. It might raise issues regarding the pickle files if you evaluated without training

python ../lsf3d/inference/predict_simple.py -i ../litesegformer3d/DATASET_Tumor/litesegformer3d_raw/litesegformer3d_raw_data/Task003_tumor/imagesTs -o ../litesegformer3d/litesegformer3d/evaluation/litesegformer3d_tumor_checkpoint/inferTs -m 3d_fullres  -t 3 -f 0 -chk model_final_checkpoint -tr litesegformer3d_trainer_tumor

python ../lsf3d/inference_tumor.py 0

