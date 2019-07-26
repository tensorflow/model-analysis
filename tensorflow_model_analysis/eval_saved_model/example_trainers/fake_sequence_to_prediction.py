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
"""Exports a simple fake sequence to prediction model.

The inputs are values_t1, values_t2, values_t3, which are floats that either
absent, or contain one value.

These are converted to three-dimensional "embedding" features,
embedding_t1, embedding_t2, embedding_t3, which are [1, 1, 1] * values_t,
or [0 0 0] if the value is absent. These are then combined into a single
"embedding" feature.

We also create a "sparse_values" feature from the values, whose dense_value is
[values_t, values_t ** 2, values_t ** 3] for each timestep (in practice this
would be better represented as dense tensor, but we make it sparse tensor so
we can exercise the sparse tensor codepaths).

We do this to simulate a sequence model where the features have dimensions
[batch_size, time_step, feature_legth] to exercise the case where the features
(both dense and sparse) have more than 2 dimensions.

For example, given:
  values_t1 = [1, 10], values_t2 = [2, 20], values_t3 = [3, 30],

we have:
  embedding[0, 0, :] = [1 1 1]
  embedding[1, 0, :] = [10 10 10]
  embedding[0, 1, :] = [2 2 2]
  embedding[1, 1, :] = [20 20 20]
  embedding[0, 2, :] = [3 3 3]
  embedding[1, 2, :] = [30 30 30]

and (after converting to dense tensor):
  sparse_values[0, 0, :] = [1 1 1]
  sparse_values[1, 0, :] = [10 100 1000]
  sparse_values[0, 1, :] = [2 4 8]
  sparse_values[1, 1, :] = [20 400 8000]
  sparse_values[0, 2, :] = [3 9 27]
  sparse_values[1, 2, :] = [30 900 27000]

The model is parameterised as:
   a * sum(embedding[:,0,:])
 + b * sum(embedding[:,1,:])
 + c * sum(embedding[:,2,:])
 + d * sum(sparse_values[:, 0, :])
 + e * sum(sparse_values[:, 1, :])
 + f * sum(sparse_values[:, 2, :])

which given how the "embeddings" and sparse values are generated is really just:
   3a * values_t1 + 3b * values_t2 + 3c * values_t3
 +  d * (values_t1 + values_t1 ** 2 + values_t1 ** 3)
 +  e * (values_t2 + values_t2 ** 2 + values_t3 ** 3)
 +  f * (values_t2 + values_t2 ** 2 + values_t3 ** 3)

The data was generated using a=1, b=2, c=3, d=4, e=5, f=6 and we initialize
the parameters to these values, so the model doesn't actually do any training,
and produces the prediction:
   3 * values_t1 + 6 * values_t2 + 9 * values_t3
 + 4 * (values_t1 + values_t1 ** 2 + values_t1 ** 3)
 + 5 * (values_t2 + values_t2 ** 2 + values_t2 ** 3)
 + 6 * (values_t3 + values_t3 ** 2 + values_t3 ** 3)
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model import util


def simple_fake_sequence_to_prediction(export_path, eval_export_path):
  """Trains and exports a fake_sequence_to_prediction model."""

  input_feature_spec = {
      'values_t1': tf.io.VarLenFeature(dtype=tf.float32),
      'values_t2': tf.io.VarLenFeature(dtype=tf.float32),
      'values_t3': tf.io.VarLenFeature(dtype=tf.float32)
  }
  label_feature_spec = dict(input_feature_spec)
  label_feature_spec['label'] = tf.io.FixedLenFeature([1], dtype=tf.float32)

  def _make_embedding_and_sparse_values(features):
    """Make "embedding" and "sparse_values" features."""
    embedding_dim = 3
    sparse_dims = 3
    sparse_timesteps = 3

    # Create a three-dimensional "embedding" based on the value of the feature
    # The embedding is simply [1, 1, 1] * feature_value
    # (or [0, 0, 0] if the feature is missing).
    batch_size = tf.cast(
        tf.shape(input=features['values_t1'])[0], dtype=tf.int64)

    ones = tf.ones(shape=[embedding_dim])
    dense_t1 = tf.sparse.to_dense(features['values_t1'])
    dense_t2 = tf.sparse.to_dense(features['values_t2'])
    dense_t3 = tf.sparse.to_dense(features['values_t3'])
    embedding_t1 = ones * dense_t1
    embedding_t2 = ones * dense_t2
    embedding_t3 = ones * dense_t3
    embeddings = tf.stack([embedding_t1, embedding_t2, embedding_t3], axis=1)
    features['embedding'] = embeddings
    del features['values_t1']
    del features['values_t2']
    del features['values_t3']

    # Make the "sparse_values" feature.
    sparse_values = tf.squeeze(
        tf.concat([
            dense_t1, dense_t1**2, dense_t1**3, dense_t2, dense_t2**2, dense_t2
            **3, dense_t3, dense_t3**2, dense_t3**3
        ],
                  axis=0))
    sparse_total_elems = batch_size * sparse_dims * sparse_timesteps
    seq = tf.range(0, sparse_total_elems, dtype=tf.int64)
    batch_num = seq % batch_size
    timestep = tf.compat.v1.div(seq, batch_size * sparse_dims)
    offset = tf.compat.v1.div(seq, batch_size) % sparse_dims
    sparse_indices = tf.stack([batch_num, timestep, offset], axis=1)
    features['sparse_values'] = tf.SparseTensor(
        indices=sparse_indices,
        values=sparse_values,
        dense_shape=[batch_size, sparse_timesteps, sparse_dims])

  def model_fn(features, labels, mode, config):
    """Model function for custom estimator."""
    del config
    dense_values = tf.sparse.to_dense(
        features['sparse_values'], validate_indices=False)
    a = tf.Variable(1.0, dtype=tf.float32, name='a')
    b = tf.Variable(2.0, dtype=tf.float32, name='b')
    c = tf.Variable(3.0, dtype=tf.float32, name='c')
    d = tf.Variable(4.0, dtype=tf.float32, name='d')
    e = tf.Variable(5.0, dtype=tf.float32, name='e')
    f = tf.Variable(6.0, dtype=tf.float32, name='f')
    predictions = (
        a * tf.reduce_sum(input_tensor=features['embedding'][:, 0, :], axis=1) +
        b * tf.reduce_sum(input_tensor=features['embedding'][:, 1, :], axis=1) +
        c * tf.reduce_sum(input_tensor=features['embedding'][:, 2, :], axis=1) +
        d * tf.reduce_sum(input_tensor=dense_values[:, 0, :], axis=1) +
        e * tf.reduce_sum(input_tensor=dense_values[:, 1, :], axis=1) +
        f * tf.reduce_sum(input_tensor=dense_values[:, 2, :], axis=1))

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={'score': predictions},
          export_outputs={
              'score': tf.estimator.export.RegressionOutput(predictions)
          })

    loss = tf.compat.v1.losses.mean_squared_error(
        labels, tf.expand_dims(predictions, axis=-1))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=0.0001)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            'mean_squared_error':
                tf.compat.v1.metrics.mean_squared_error(
                    labels, tf.expand_dims(predictions, axis=-1)),
            'mean_prediction':
                tf.compat.v1.metrics.mean(predictions),
        },
        predictions=predictions)

  def train_input_fn():
    """Train input function."""

    def make_example_with_label(values_t1=None, values_t2=None, values_t3=None):
      """Make example with label."""
      effective_t1 = 0.0
      effective_t2 = 0.0
      effective_t3 = 0.0
      args = {}
      if values_t1 is not None:
        args['values_t1'] = float(values_t1)
        effective_t1 = values_t1
      if values_t2 is not None:
        args['values_t2'] = float(values_t2)
        effective_t2 = values_t2
      if values_t3 is not None:
        args['values_t3'] = float(values_t3)
        effective_t3 = values_t3
      label = (3 * effective_t1 + 6 * effective_t2 + 9 * effective_t3 + 4 *
               (effective_t1 + effective_t1**2 + effective_t1**3) + 5 *
               (effective_t2 + effective_t2**2 + effective_t2**3) + 6 *
               (effective_t3 + effective_t3**2 + effective_t3**3))
      args['label'] = float(label)
      return util.make_example(**args)

    examples = [
        make_example_with_label(values_t1=1.0),
        make_example_with_label(values_t2=1.0),
        make_example_with_label(values_t3=1.0),
        make_example_with_label(values_t1=2.0, values_t2=3.0),
        make_example_with_label(values_t1=5.0, values_t3=7.0),
        make_example_with_label(values_t2=11.0, values_t3=13.0),
        make_example_with_label(values_t1=2.0, values_t2=3.0, values_t3=5.0),
    ]
    serialized_examples = [x.SerializeToString() for x in examples]
    features = tf.io.parse_example(
        serialized=serialized_examples, features=label_feature_spec)
    _make_embedding_and_sparse_values(features)
    label = features.pop('label')
    return features, label

  def serving_input_receiver_fn():
    """Serving input receiver function."""
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.io.parse_example(
        serialized=serialized_tf_example, features=input_feature_spec)
    _make_embedding_and_sparse_values(features)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def eval_input_receiver_fn():
    """Eval input receiver function."""
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.io.parse_example(
        serialized=serialized_tf_example, features=label_feature_spec)
    _make_embedding_and_sparse_values(features)

    return export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=features['label'])

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn, steps=10)

  export_dir = None
  eval_export_dir = None
  if export_path:
    export_dir = estimator.export_saved_model(
        export_dir_base=export_path,
        serving_input_receiver_fn=serving_input_receiver_fn)

  if eval_export_path:
    eval_export_dir = export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=eval_export_path,
        eval_input_receiver_fn=eval_input_receiver_fn)

  return export_dir, eval_export_dir
