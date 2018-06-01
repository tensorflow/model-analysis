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
FROM ubuntu:18.04

MAINTAINER TFX <tensorflow-extended-dev@googlegroups.com>

RUN apt-get update && apt-get install -y \
        curl \
        gnupg

# Add TensorFlow Serving repo

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN curl -s https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

# Get the model-server, use tensorflow-model-server-universal for older hardware
RUN apt-get update && apt-get install -y \
        tensorflow-model-server

# Cleanup to reduce the size of the image
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN  apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Create model directory

ENV MODEL_DIR /model
ENV MODEL_NAME chicago_taxi
ENV MODEL_PORT 9000

RUN mkdir ${MODEL_DIR}
WORKDIR ${MODEL_DIR}  # We will start the Model Server here

ENV MODEL_SERVER_CMD "/usr/bin/tensorflow_model_server --port=${MODEL_PORT} --model_base_path=${MODEL_DIR} --model_name=${MODEL_NAME}"

CMD $MODEL_SERVER_CMD
