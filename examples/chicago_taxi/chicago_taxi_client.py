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
"""A client for the chicago_taxi demo."""

from __future__ import print_function

import argparse
from grpc.beta import implementations

import six
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from trainer import taxi

_TIMEOUT_SECONDS = 5.0


def _do_inference(hostport, examples_file, num_examples):
  """Sends a request to the model and returns the result.

  Args:
    hostport: path to prediction service like host:port
    examples_file: path to csv file containing examples, with the first line
      assumed to have the column headers
    num_examples: number of requests to send to the server

  Returns:
    Response from model server
  """
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  csv_coder = taxi.make_csv_coder()
  f = open(examples_file, 'r')
  f.readline()  # skip header line

  for _ in range(num_examples):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'chicago_taxi'
    request.model_spec.signature_name = 'predict'
    one_line = f.readline()
    if not one_line:
      print('End of example file reached')
      return

    one_example = taxi.clean_raw_data_dict(csv_coder.decode(one_line))
    print(one_example)

    raw_feature_spec = taxi.get_raw_feature_spec()
    for key, val in six.iteritems(one_example):
      if key != 'tips':
        tfproto = tf.contrib.util.make_tensor_proto(
            val, shape=[1], dtype=raw_feature_spec[key].dtype)
        request.inputs[key].CopyFrom(tfproto)

    return stub.Predict(request, _TIMEOUT_SECONDS)


def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_examples',
      help=('Number of examples to send to the server.'),
      default=1,
      type=int)

  parser.add_argument(
      '--server',
      help=('Prediction service host:port'),
      required=True)

  parser.add_argument(
      '--examples_file',
      help=('Path to csv file containing examples.'),
      required=True)

  known_args, _ = parser.parse_known_args()
  result = _do_inference(known_args.server,
                         known_args.examples_file,
                         known_args.num_examples)
  print(result)


if __name__ == '__main__':
  tf.app.run()
