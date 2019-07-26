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
"""Exports a "fixed prediction" model that treats 1 input as 0 to n examples.

This model takes in number of examples as input, and predicts the "input_index"
for each example.

This model has 4 features, "example_count", "input_index" and
"intra_input_index" and "annotations". Its prediction equals "input_index".
If the input numbers of examples (as a batch) is [2, 0, 3],
then example_count = [2, 2, 3, 3, 3], input_index = [0, 0, 1, 1, 1]
and intra_input_index = [0, 1, 0, 1, 2].

In this file also are functions to export models with bad input_refs to test
error handling in TFMA.

This is to test that TFMA can handle the case where raw_input_bytes:examples is
m:n (usually m=n=1).
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model.example_trainers import util
from tensorflow.python.estimator.canned import metric_keys


def _indices_from_example_count(example_count):
  """Computes input indices from example count.

  Args:
    example_count: a 1-D tf.Tensor, representing a batch of inputs. Each element
      is the number of examples that input corresponds to.

  Returns:
    A tuple of input_index and intra_input_index. For an example,
    input_index = i means it came from the i-th input (i.e. example_count[i]).
    intra_input_index = i means it is the i-th among all the examples from
    the same input.
  """
  total_num_examples = tf.reduce_sum(input_tensor=example_count)
  example_indices = tf.range(total_num_examples)
  input_limits = tf.cumsum(example_count)
  index_less_than_limit = (
      tf.expand_dims(example_indices, 0) >= tf.expand_dims(input_limits, 1))
  input_index = tf.reduce_sum(
      input_tensor=tf.cast(index_less_than_limit, dtype=tf.int32), axis=0)

  offset = tf.cumsum(
      tf.concat([tf.constant([0], tf.int32), example_count], axis=0)[:-1])
  intra_input_index = example_indices - tf.gather(offset, input_index)
  return input_index, intra_input_index


def _parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""
  example_count = tf.io.decode_csv(
      records=rows_string_tensor,
      record_defaults=[tf.constant([0], dtype=tf.int32, shape=None)])[0]

  input_index, intra_input_index = _indices_from_example_count(example_count)
  annotation = tf.strings.join([
      'raw_input: ',
      tf.gather(rows_string_tensor, input_index), '; index: ',
      tf.as_string(intra_input_index)
  ])

  return {
      'example_count': tf.gather(example_count, input_index),
      'input_index': input_index,
      'intra_input_index': intra_input_index,
      'annotation': annotation,
  }


def _eval_input_receiver_fn():
  """Eval input receiver function."""
  csv_row = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_csv_row')
  features = _parse_csv(csv_row)
  receiver_tensors = {'examples': csv_row}

  return export.EvalInputReceiver(
      features=features,
      labels=features['input_index'],
      receiver_tensors=receiver_tensors,
      input_refs=features['input_index'])


def _eval_input_receiver_using_iterator_fn():
  """Eval input receiver function using an iterator."""
  csv_row = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_csv_row')
  iterator = tf.compat.v1.data.make_initializable_iterator(
      tf.compat.v1.data.Dataset.from_tensors(csv_row))
  features = _parse_csv(iterator.get_next())
  receiver_tensors = {'examples': csv_row}

  return export.EvalInputReceiver(
      features=features,
      labels=features['input_index'],
      receiver_tensors=receiver_tensors,
      input_refs=features['input_index'],
      iterator_initializer=iterator.initializer.name)


def _legacy_eval_input_receiver_fn():
  """Legacy eval input receiver function."""
  csv_row = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_csv_row')
  features = _parse_csv(csv_row)
  receiver_tensors = {'examples': csv_row}

  # the constructor of _LegacyEvalInputReceiver() has side effects (populating
  # some TF collections). Calling twice here to make sure the collisions are
  # handled correctly.
  export._LegacyEvalInputReceiver(  # pylint: disable=protected-access
      features=features,
      labels=features['input_index'],
      receiver_tensors=receiver_tensors,
      input_refs=features['input_index'])

  return export._LegacyEvalInputReceiver(  # pylint: disable=protected-access
      features=features,
      labels=features['input_index'],
      receiver_tensors=receiver_tensors,
      input_refs=features['input_index'])


def _bad_eval_input_receiver_fn_misaligned_input_refs():
  """A bad eval input receiver function capturing a misaligned input_refs."""
  csv_row = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_csv_row')
  features = _parse_csv(csv_row)
  receiver_tensors = {'examples': csv_row}

  return export.EvalInputReceiver(
      features=features,
      labels=features['input_index'],
      receiver_tensors=receiver_tensors,
      input_refs=tf.concat(
          [features['input_index'],
           tf.constant([0], dtype=tf.int32)], axis=0))


def _bad_eval_input_receiver_fn_out_of_range_input_refs():
  """A bad eval input receiver function (input_refs has out-of-range index)."""
  csv_row = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_csv_row')
  features = _parse_csv(csv_row)
  receiver_tensors = {'examples': csv_row}

  return export.EvalInputReceiver(
      features=features,
      labels=features['input_index'],
      receiver_tensors=receiver_tensors,
      input_refs=features['input_index'] + 1)


def _serving_input_receiver_fn():
  """Serving input receiver function."""
  csv_row = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_csv_row')
  features = _parse_csv(csv_row)
  receiver_tensors = {'examples': csv_row}
  return tf.estimator.export.ServingInputReceiver(
      features=features, receiver_tensors=receiver_tensors)


def _train_input_fn():
  """Train input function."""
  features = (
      tf.compat.v1.data.make_one_shot_iterator(
          tf.data.Dataset.from_tensors(tf.constant(
              ['3', '0', '1', '2'])).repeat().map(_parse_csv)).get_next())

  return features, features['input_index']


def _model_fn(features, labels, mode, config):
  """Model function for custom estimator."""

  del config  # Unused.

  predictions = tf.cast(features['input_index'], tf.float32)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.RegressionOutput(predictions)
        })

  loss = tf.compat.v1.losses.mean_squared_error(features['example_count'],
                                                labels)
  train_op = tf.compat.v1.assign_add(tf.compat.v1.train.get_global_step(), 1)
  eval_metric_ops = {
      metric_keys.MetricKeys.LOSS_MEAN: tf.compat.v1.metrics.mean(loss),
  }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions=predictions,
      eval_metric_ops=eval_metric_ops)


def fake_multi_examples_per_input_estimator(export_path,
                                            eval_export_path,
                                            use_iterator=False):
  """Trains and exports a model that treats 1 input as 0 to n examples ."""
  estimator = tf.estimator.Estimator(model_fn=_model_fn)
  estimator.train(input_fn=_train_input_fn, steps=1)

  eval_input_receiver_fn = _eval_input_receiver_fn
  if use_iterator:
    eval_input_receiver_fn = _eval_input_receiver_using_iterator_fn
  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=_serving_input_receiver_fn,
      eval_input_receiver_fn=eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)


def legacy_fake_multi_examples_per_input_estimator(export_path,
                                                   eval_export_path):
  """Trains and exports a model that treats 1 input as 0 to n examples ."""
  estimator = tf.estimator.Estimator(model_fn=_model_fn)
  estimator.train(input_fn=_train_input_fn, steps=1)

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=_serving_input_receiver_fn,
      eval_input_receiver_fn=_legacy_eval_input_receiver_fn,
      export_path=export_path,
      eval_export_path=eval_export_path)


def bad_multi_examples_per_input_estimator_misaligned_input_refs(
    export_path, eval_export_path):
  """Like the above (good) estimator, but the input_refs is misaligned."""
  estimator = tf.estimator.Estimator(model_fn=_model_fn)
  estimator.train(input_fn=_train_input_fn, steps=1)

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=_serving_input_receiver_fn,
      eval_input_receiver_fn=_bad_eval_input_receiver_fn_misaligned_input_refs,
      export_path=export_path,
      eval_export_path=eval_export_path)


def bad_multi_examples_per_input_estimator_out_of_range_input_refs(
    export_path, eval_export_path):
  """Like the above (good) estimator, but the input_refs is out of range."""
  estimator = tf.estimator.Estimator(model_fn=_model_fn)
  estimator.train(input_fn=_train_input_fn, steps=1)

  return util.export_model_and_eval_model(
      estimator=estimator,
      serving_input_receiver_fn=_serving_input_receiver_fn,
      eval_input_receiver_fn=(
          _bad_eval_input_receiver_fn_out_of_range_input_refs),
      export_path=export_path,
      eval_export_path=eval_export_path)
