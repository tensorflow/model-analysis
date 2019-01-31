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

echo Starting flink TFT preprocessing...

if [ -z "$MYBUCKET" ]; then
  echo MYBUCKET was not set
  echo Please set MYBUCKET to your GCP bucket using: export MYBUCKET=gs://bucket
  exit 1
fi

JOB_ID="chicago-taxi-preprocess-$(date +%Y%m%d-%H%M%S)"
JOB_INPUT_PATH=$MYBUCKET/$JOB_ID/chicago_taxi_input
JOB_OUTPUT_PATH=$MYBUCKET/$JOB_ID/chicago_taxi_output
export TFT_OUTPUT_PATH=$JOB_OUTPUT_PATH/tft_output
TEMP_PATH=$MYBUCKET/$JOB_ID/tmp/
MYPROJECT=$(gcloud config list --format 'value(core.project)' 2>/dev/null)
SCHEMA_PATH=./data/flink_tfdv_output/schema.pbtxt

echo Using GCP project: $MYPROJECT
echo Job input path: $JOB_INPUT_PATH
echo Job output path: $JOB_OUTPUT_PATH
echo TFT output path: $TFT_OUTPUT_PATH

: << 'END'
JOB_ID="chicago-taxi-preprocess-$(date +%Y%m%d-%H%M%S)"
JOB_INPUT_PATH=$(pwd)/data
JOB_OUTPUT_PATH=$(pwd)/flink_output
export TFT_OUTPUT_PATH=$JOB_OUTPUT_PATH/tft_output
TEMP_PATH=/tmp/flink-tfx
MYPROJECT=$(gcloud config list --format 'value(core.project)' 2>/dev/null)

echo Using GCP project: $MYPROJECT
echo Job input path: $JOB_INPUT_PATH
echo Job output path: $JOB_OUTPUT_PATH
echo TFT output path: $TFT_OUTPUT_PATH
END

# move data to gcs
echo Uploading data to GCS
gsutil cp -r ./data/eval/ ./data/train/ $JOB_INPUT_PATH/


# Preprocess the eval files
echo Preprocessing eval data...
rm -R -f $(pwd)/data/eval/local_chicago_taxi_output

#image="$(whoami)-docker-apache.bintray.io/beam/python"
image="gcr.io/dataflow-build/goenka/beam_fnapi_python"


#input=gs://clouddfe-goenka/chicago_taxi_data/taxi_trips_000000000000.csv
input=$JOB_INPUT_PATH/eval/data.csv
#input=$JOB_INPUT_PATH/eval/data_medium.csv
#input=$JOB_INPUT_PATH/eval/data_133M.csv


threads=100
#sdk=--sdk_location=/usr/local/google/home/goenka/d/work/beam/beam/sdks/python/build/apache-beam-2.9.0.dev0.tar.gz
sdk=""

#extra_args="--retain_docker_containers=true"
extra_args=""

python preprocess.py \
  --output_dir $JOB_OUTPUT_PATH/eval/local_chicago_taxi_output \
  --schema_file $SCHEMA_PATH \
  --outfile_prefix eval_transformed \
  --input $input \
  --setup_file ./setup.py \
  --experiments=beam_fn_api \
  --runner PortableRunner \
  --job_endpoint=localhost:8099 \
  --experiments=worker_threads=$threads \
  $sdk \
  $extra_args \
  --environment_type=DOCKER \
  --environment_config=$image \
  --execution_mode_for_batch=BATCH_FORCED


# Preprocess the train files, keeping the transform functions
echo Preprocessing train data...
rm -R -f $(pwd)/data/train/local_chicago_taxi_output
python preprocess.py \
  --output_dir $JOB_OUTPUT_PATH/train/local_chicago_taxi_output \
  --schema_file $SCHEMA_PATH \
  --outfile_prefix train_transformed \
  --input $JOB_INPUT_PATH/train/data.csv \
  --setup_file ./setup.py \
  --experiments=beam_fn_api \
  --runner PortableRunner \
  --job_endpoint=localhost:8099 \
  --experiments=worker_threads=$threads \
  $sdk \
  $extra_args \
  --environment_type=DOCKER \
  --environment_config=$image \
  --execution_mode_for_batch=BATCH_FORCED

