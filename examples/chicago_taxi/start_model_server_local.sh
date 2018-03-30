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

if ! [ -x "$(command -v docker)" ]; then
    echo "ERROR: This script requires Docker"
    exit 1
fi

# This script first builds a Docker image and then runs it. The name of the
# image to build and run is specified by the constant below
DOCKER_IMAGE_NAME=tfma

echo Building the Docker image: $DOCKER_IMAGE_NAME

docker build -t $DOCKER_IMAGE_NAME .
# Dir for model exported for serving
LOCAL_MODEL_DIR=$(pwd)/data/train/local_chicago_taxi_output/serving_model_dir/export/chicago-taxi
# Make sure the trained model is available
if ! [ -d "$LOCAL_MODEL_DIR" ]; then
  echo "ERROR: Could not find the exported model directory $LOCAL_MODEL_DIR"
  exit 1
fi

echo Starting the Model Server to serve from: $LOCAL_MODEL_DIR

# Container model dir
CONTAINER_MODEL_DIR=/model

# Local port where to send inference requests
HOST_PORT=9000

# Where our contianer is listening
CONTAINER_PORT=9000

echo Model directory: $LOCAL_MODEL_DIR

docker run -it\
  -p 127.0.0.1:$HOST_PORT:$CONTAINER_PORT \
  -v $LOCAL_MODEL_DIR:$CONTAINER_MODEL_DIR \
  --rm $DOCKER_IMAGE_NAME
