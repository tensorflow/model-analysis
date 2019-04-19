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
"""Library handling the model agnostic TensorFlow graph.

Model Agnostic Prediction is the flow to generate FeaturesPredictionLabels
when the training eval model is not available. Currently, this flow supports
converting tf.Example protos to FPL provided an explicit key -> [F,P,L] mapping
and a parsing spec. This represents the minimum amount of information needed
to derive FeaturesPredictionLabels. This feature is useful when a user wants to
run tf.Metrics or postExportMetrics when the training eval model is not
available.

An example set of inputs is:

  tf.Example{ features {
      feature {
        key: "age" value { float_list { value: 29.0 } } }
      feature {
        key: "language" value { bytes_list { value: "English" } } }
      feature {
        key: "predictions" value { float_list { value: 1.0 } } }
      feature {
        key: "labels" value { float_list { value: 2.0 } } }
    }
  }

  feature_spec = {
      'age':
          tf.FixedLenFeature([], tf.float32),
      'language':
          tf.VarLenFeature(tf.string),
      'predictions':
          tf.FixedLenFeature([], tf.float32),
      'labels':
          tf.FixedLenFeature([], tf.float32)
  }

  model_agnostic_config = model_agnostic_predict.ModelAgnosticConfig(
      label_keys=['labels'],
      prediction_keys=['predictions'],
      feature_spec=feature_spec)

Then the expected output is:

  FPL.features = {'age' : np.array[29.0],
                  'language': SparseTensorValue('English')}
  FPL.predictions = {'predictions' : np.array[1.0]}
  FPL.labels = {'labels' : np.array[2.0]}
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util as general_util
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import util

from typing import Any, Dict, List, NamedTuple, Text  # pytype: disable=not-supported-yet


class ModelAgnosticConfig(
    NamedTuple(  # pylint: disable=invalid-name
        'ModelAgnosticConfig', [
            ('label_keys', List[Text]),
            ('prediction_keys', List[Text]),
            ('feature_spec', Dict[Text, Any]),
        ])):
  """A config spec for running ModelAgnostic evaluation."""

  def __new__(cls, label_keys: List[Text], prediction_keys: List[Text],
              feature_spec: Dict[Text, Any]):
    """Creates a ModelAgnosticConfig instance.

    Creates a config spec for doing ModelAgnostic evaluation (Model evaluation
    without the training eval saved model). This spec defines the basic
    parameters with which to define Features, Predictions, and Labels from
    input Examples.

    Args:
      label_keys: A list of Text, the keys in the input examples which should be
        treated as labels. Currently, this cannot be empty.
      prediction_keys: A list of Text, the keys in the input examples which
        should be treated as predictions. Currently, this cannot be empty.
      feature_spec: In the case only FPL is provided (via Examples), a dict
        defining how to parse the example. This should be of the form "key" ->
        FixedLenFeature or VarLenFeature. This is required to parse input
        examples.

    Returns:
      A ModelAgnosticConfig instance.

    Raises:
      ValueError: This inputs supplied are properly defined..
    """

    if not label_keys:
      raise ValueError('ModelAgnosticConfig must have label keys set.')
    if not prediction_keys:
      raise ValueError('ModelAgnosticConfig must have prediction keys set.')
    if not feature_spec:
      raise ValueError('ModelAgnosticConfig must have feature_spec set.')
    for key in prediction_keys:
      if key not in feature_spec:
        raise ValueError('Prediction key %s not defined in feature_spec.' % key)
    for key in label_keys:
      if key not in feature_spec:
        raise ValueError('Label key %s not defined in feature_spec.' % key)

    return super(ModelAgnosticConfig, cls).__new__(
        cls,
        label_keys=label_keys,
        prediction_keys=prediction_keys,
        feature_spec=feature_spec)


class ModelAgnosticPredict(object):
  """Abstraction for using a model agnostic evaluation.

  This class is an API interface to interact with the with Model Agnostic graph
  to do evaluation without needing an eval_saved_model.
  It serves two primary functions:
    1) Be able to generate an FPL given FPLs encoded in the tf.Examples input.
    2) Be able to do metric evaluations against the FPLs generated.

  Design Doc: go/model-agnostic-tfma
  """

  def __init__(self, model_agnostic_config: ModelAgnosticConfig):
    self._graph = tf.Graph()
    self._session = tf.compat.v1.Session(graph=self._graph)
    self._config = model_agnostic_config
    try:
      self._create_graph()
    except (RuntimeError, ValueError) as exception:
      general_util.reraise_augmented(exception,
                                     'Failed to initialize agnostic model')

  def _create_graph(self):
    """Creates the graph for which we use to generate FPL and metrics.

    Create a pass-through graph which parses the input examples using the
    feature spec.
    """
    with self._graph.as_default():
      serialized_example = tf.compat.v1.placeholder(dtype=tf.string)
      features = tf.io.parse_example(
          serialized=serialized_example, features=self._config.feature_spec)
      self._get_features_fn = self._session.make_callable(
          fetches=features, feed_list=[serialized_example])

  def get_fpls_from_examples(self, input_example_bytes_list: List[bytes]
                            ) -> List[Any]:
    """Generates FPLs from serialized examples using a ModelAgnostic graph.

    Args:
      input_example_bytes_list: A string representing the serialized tf.example
        protos to be parsed by the graph.

    Returns:
      A list of FeaturesPredictionsLabels generated from the input examples.
    """
    # Call the graph via the created session callable _get_features_fn and
    # get the tensor representation of the features.
    features = self._get_features_fn(input_example_bytes_list)
    split_features = {}
    num_examples = 0

    # Split the features by the example keys. Also verify all each example
    # key has the same number of total examples.
    for key in features.keys():
      split_features[key] = util.split_tensor_value(features[key])
      if num_examples == 0:
        num_examples = len(split_features[key])
      elif num_examples != len(split_features[key]):
        raise ValueError(
            'Different keys unexpectedly had different number of '
            'examples. Key %s unexpectedly had %s elements.' % key,
            len(split_features[key]))

    # Sort out the examples into individual FPLs: one example -> one FPL.
    # Sort them into Features, Predictions, or Labels according to the input
    # config.
    result = []
    for i in range(num_examples):
      labels = {}
      predictions = {}
      features = {}
      for key in split_features:
        if key in self._config.label_keys:
          labels[key] = {encoding.NODE_SUFFIX: split_features[key][i]}
        if key in self._config.prediction_keys:
          predictions[key] = {encoding.NODE_SUFFIX: split_features[key][i]}
        features[key] = {encoding.NODE_SUFFIX: split_features[key][i]}

      result.append(
          types.FeaturesPredictionsLabels(
              input_ref=i,
              features=features,
              predictions=predictions,
              labels=labels))

    return result
