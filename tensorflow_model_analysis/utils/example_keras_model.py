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
"""Example Keras model classes for testing and demonstration purposes."""

from tensorflow import keras
import tensorflow.compat.v1 as tf
import tensorflow_model_analysis as tfma


AGE = 'age'
LANGUAGE = 'language'
LABEL = 'label'
SLICE = 'my_slice'
FEATURE_MAP = {
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    AGE: tf.io.FixedLenFeature([], tf.float32),
    LANGUAGE: tf.io.FixedLenFeature([], tf.string),
    SLICE: tf.io.VarLenFeature(tf.string),
}


class ExampleCLassifierParser(keras.layers.Layer):
  """A Keras layer that parses the tf.Example."""

  def __init__(self, input_feature_key):
    self._input_feature_key = input_feature_key
    super().__init__()

  def call(self, serialized_examples):
    def get_feature(serialized_example):
      parsed_example = tf.io.parse_single_example(
          serialized_example, features=FEATURE_MAP
      )
      return parsed_example[self._input_feature_key]

    return tf.map_fn(get_feature, serialized_examples)


class ExampleClassifierModel(keras.Model):
  """A Example Keras NLP model."""

  def __init__(self, input_feature_key: str = LANGUAGE):
    super().__init__()
    self.parser = ExampleCLassifierParser(input_feature_key)
    self.text_vectorization = keras.layers.TextVectorization(
        max_tokens=32,
        output_mode='int',
        output_sequence_length=32,
    )
    self.text_vectorization.adapt([
        'nontoxic',
        'toxic comment',
        'japanese',
        'hindi',
        'english',
        'chinese',
        'abcdef',
        'random',
    ])
    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(1)

  def call(self, inputs, training=True, mask=None):
    parsed_example = self.parser(inputs)
    text_vector = self.text_vectorization(parsed_example)
    output1 = self.dense1(tf.cast(text_vector, tf.float32))
    output2 = self.dense2(output1)
    return output2


def evaluate_model(
    classifier_model_path,
    validate_tf_file_path,
    tfma_eval_result_path,
    eval_config,
):
  """Evaluate Model using Tensorflow Model Analysis.

  Args:
    classifier_model_path: Trained classifier model to be evaluted.
    validate_tf_file_path: File containing validation TFRecordDataset.
    tfma_eval_result_path: Path to export tfma-related eval path.
    eval_config: tfma eval_config.
  """

  eval_shared_model = tfma.default_eval_shared_model(
      eval_saved_model_path=classifier_model_path, eval_config=eval_config
  )

  # Run the fairness evaluation.
  tfma.run_model_analysis(
      eval_shared_model=eval_shared_model,
      data_location=validate_tf_file_path,
      output_path=tfma_eval_result_path,
      eval_config=eval_config,
  )
