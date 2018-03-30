#!/bin/bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -u

echo Starting local training...

# Output: dir for our raw=>transform function
WORKING_DIR=$(pwd)/data/train/local_chicago_taxi_output

# Output: dir for both the serving model and eval_model which will go into tfma
# evaluation
OUTPUT_DIR=$WORKING_DIR
rm -R -f $OUTPUT_DIR/serving_model_dir
rm -R -f $OUTPUT_DIR/eval_model_dir

# Output: dir for trained model
MODEL_DIR=$WORKING_DIR/trainer_output
rm -R -f $MODEL_DIR

echo Working directory: $WORKING_DIR
echo Serving model directory: $OUTPUT_DIR/serving_model_dir
echo Eval model directory: $OUTPUT_DIR/eval_model_dir

# Inputs: are tf-transformed materialized features
TRAIN_FILE=$(pwd)/data/train/local_chicago_taxi_output/train_transformed-*
EVAL_FILE=$(pwd)/data/eval/local_chicago_taxi_output/eval_transformed-*

# Parameters for tf training and evaluation (not tfma)
TRAIN_STEPS=10000
EVAL_STEPS=5000

python trainer/task.py \
    --train-files $TRAIN_FILE \
    --verbosity=INFO \
    --job-dir $MODEL_DIR \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --tf-transform-dir $WORKING_DIR \
    --output-dir $OUTPUT_DIR \
    --eval-files $EVAL_FILE
