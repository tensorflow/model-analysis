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
"""API for Tensorflow Model Analysis."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tempfile

import apache_beam as beam
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.api.impl import evaluate
from tensorflow_model_analysis.api.impl import serialization
from tensorflow_model_analysis.eval_saved_model.post_export_metrics import post_export_metrics
import tensorflow_model_analysis.eval_saved_model.post_export_metrics.metric_keys as metric_keys
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Any, Dict, List, Optional
from google.protobuf import json_format


# The input type is a MessageMap where the keys are strings and the values are
# some protocol buffer field. Note that MessageMap is not a protobuf message,
# none of the exising utility methods work on it. We must iterate over its
# values and call the utility function individually.
def _convert_metric_map_to_dict(
    metric_map):
  """Converts a metric map (metrics in MetricsForSlice protobuf) into a dict.

  Args:
    metric_map: A protocol buffer MessageMap that has behaviors like dict. The
      keys are strings while the values are protocol buffers. However, it is not
      a protobuf message and cannot be passed into json_format.MessageToDict
      directly. Instead, we must iterate over its values.

  Returns:
    A dict representing the metric_map. For example:
    Assume myProto contains
    {
      metrics: {
        key: 'double'
        value: {
          double_value: {
            value: 1.0
          }
        }
      }
      metrics: {
        key: 'bounded'
        value: {
          bounded_value: {
            lower_bound: {
              double_value: {
                value: 0.8
              }
            }
            upper_bound: {
              double_value: {
                value: 0.9
              }
            }
            value: {
              double_value: {
                value: 0.86
              }
            }
          }
        }
      }
    }

    The output of _convert_metric_map_to_dict(myProto.metrics) would be

    {
      'double': {
        'doubleValue': 1.0,
      },
      'bounded': {
        'boundedValue': {
          'lowerBound': 0.8,
          'upperBound': 0.9,
          'value': 0.86,
        },
      },
    }

    Note that field names are converted to lowerCamelCase and the field value in
    google.protobuf.DoubleValue is collapsed automatically.
  """
  return {k: json_format.MessageToDict(metric_map[k]) for k in metric_map}


def load_eval_result(output_path):
  """Creates an EvalResult object for use with the visualization functions."""
  metrics_proto_list, plots_proto_list = serialization.load_plots_and_metrics(
      output_path)

  slicing_metrics = [(key, _convert_metric_map_to_dict(metrics_data))
                     for key, metrics_data in metrics_proto_list]
  plots = [(key, json_format.MessageToDict(plot_data))
           for key, plot_data in plots_proto_list]

  eval_config = serialization.load_eval_config(output_path)
  return api_types.EvalResult(
      slicing_metrics=slicing_metrics, plots=plots, config=eval_config)


def _assert_tensorflow_version():
  # Fail with a clear error in case we are not using a compatible TF version.
  major, minor, _ = tf.__version__.split('.')
  if int(major) != 1 or int(minor) < 10:
    raise RuntimeError(
        'Tensorflow version >= 1.10, < 2 is required. Found (%s). Please '
        'install the latest 1.x version from '
        'https://github.com/tensorflow/tensorflow. ' % tf.__version__)


@beam.ptransform_fn
@beam.typehints.with_output_types(beam.pvalue.PDone)
def EvaluateAndWriteResults(  # pylint: disable=invalid-name
    examples,
    eval_saved_model_path,
    output_path,
    display_only_data_location = None,
    slice_spec = None,
    example_weight_key = None,
    add_metrics_callbacks = None,  # pylint: disable=bad-whitespace
    desired_batch_size = None,
):
  """Public API version of evaluate.Evaluate that handles example weights.

  Users who want to construct their own Beam pipelines instead of using the
  lightweight run_model_analysis functions should use this PTransform.

  Example usage:

    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
           | 'EvaluateAndWriteResults' >> tfma.EvaluateAndWriteResults(
               eval_saved_model_path=model_location,
               output_path=output_path,
               display_only_data_location=data_location,
               slice_spec=slice_spec,
               example_weight_key=example_weight_key,
               ...))
    result = tfma.load_eval_result(output_path=output_path)
    tfma.view.render_slicing_metrics(result)

  Note that the exact serialization format is an internal implementation detail
  and subject to change. Users should only use the TFMA functions to write and
  read the results.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_saved_model_path: Path to EvalSavedModel. This directory should contain
      the saved_model.pb file.
    output_path: Path to output metrics and plots results.
    display_only_data_location: Optional path indicating where the examples
      were read from. This is used only for display purposes - data will not
      actually be read from this path.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    example_weight_key: The key of the example weight column. If None, weight
      will be 1 for each example.
    add_metrics_callbacks: Optional list of callbacks for adding additional
      metrics to the graph. The names of the metrics added by the callbacks
      should not conflict with existing metrics, or metrics added by other
      callbacks. See below for more details about what each callback should do.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.

  Returns:
    PDone.
  """

  if add_metrics_callbacks is None:
    add_metrics_callbacks = []

  # Always compute example weight and example count.
  # pytype: disable=module-attr
  example_count_callback = post_export_metrics.example_count()
  example_weight_metric_key = metric_keys.EXAMPLE_COUNT
  add_metrics_callbacks.append(example_count_callback)
  if example_weight_key:
    example_weight_metric_key = metric_keys.EXAMPLE_WEIGHT
    example_weight_callback = post_export_metrics.example_weight(
        example_weight_key)
    add_metrics_callbacks.append(example_weight_callback)
  # pytype: enable=module-attr

  metrics, plots = examples | 'Evaluate' >> evaluate.Evaluate(
      eval_saved_model_path=eval_saved_model_path,
      add_metrics_callbacks=add_metrics_callbacks,
      slice_spec=slice_spec,
      desired_batch_size=desired_batch_size)

  data_location = '<user provided PCollection>'
  if display_only_data_location is not None:
    data_location = display_only_data_location

  eval_config = api_types.EvalConfig(
      model_location=eval_saved_model_path,
      data_location=data_location,
      slice_spec=slice_spec,
      example_weight_metric_key=example_weight_metric_key)

  _ = ((metrics, plots)
       | 'SerializeMetricsAndPlots' >> serialization.SerializeMetricsAndPlots(
           post_export_metrics=add_metrics_callbacks)
       |
       'WriteMetricsPlotsAndConfig' >> serialization.WriteMetricsPlotsAndConfig(
           output_path=output_path, eval_config=eval_config))

  return beam.pvalue.PDone(examples.pipeline)


def run_model_analysis(
    model_location,
    data_location,
    file_format = 'tfrecords',
    slice_spec = None,
    example_weight_key = None,
    add_metrics_callbacks = None,  # pylint: disable=bad-whitespace
    output_path = None):
  """Runs TensorFlow model analysis.

  It runs a Beam pipeline to compute the slicing metrics exported in TensorFlow
  Eval SavedModel and returns the results.

  This is a simplified API for users who want to quickly get something running
  locally. Users who wish to create their own Beam pipelines can use the
  Evaluate PTransform instead.

  Args:
    model_location: The location of the exported eval saved model.
    data_location: The location of the data files.
    file_format: The file format of the data, can be either 'text' or
      'tfrecords' for now. By default, 'tfrecords' will be used.
    slice_spec: A list of tfma.SingleSliceSpec. Each spec represents a way to
      slice the data.
      Example usages:
      - tfma.SingleSiceSpec(): no slice, metrics are computed on overall data.
      - tfma.SingleSiceSpec(columns=['country']): slice based on features in
        column "country". We might get metrics for slice "country:us",
        "country:jp", and etc in results.
      - tfma.SingleSiceSpec(features=[('country', 'us')]): metrics are computed
        on slice "country:us".
      If None, defaults to the overall slice.
    example_weight_key: The key of the example weight column. If None, weight
      will be 1 for each example.
    add_metrics_callbacks: Optional list of callbacks for adding additional
      metrics to the graph. The names of the metrics added by the callbacks
      should not conflict with existing metrics, or metrics added by other
      callbacks. See docstring for tensorflow_model_analysis.evaluate.Evaluate
      for more details.
    output_path: The directory to output metrics and results to. If None, we use
      a temporary directory.

  Returns:
    An EvalResult that can be used with the TFMA visualization functions.

  Raises:
    ValueError: If the file_format is unknown to us.
  """
  _assert_tensorflow_version()
  # Get working_dir ready.
  if output_path is None:
    output_path = tempfile.mkdtemp()
  if not tf.gfile.Exists(output_path):
    tf.gfile.MakeDirs(output_path)

  with beam.Pipeline() as p:
    if file_format == 'tfrecords':
      data = p | 'ReadFromTFRecord' >> beam.io.ReadFromTFRecord(
          file_pattern=data_location,
          compression_type=beam.io.filesystem.CompressionTypes.UNCOMPRESSED)
    elif file_format == 'text':
      data = p | 'ReadFromText' >> beam.io.textio.ReadFromText(data_location)
    else:
      raise ValueError('unknown file_format: %s' % file_format)

    # PyLint doesn't understand Beam PTransforms.
    # pylint: disable=no-value-for-parameter
    _ = (
        data
        | 'EvaluateAndWriteResults' >> EvaluateAndWriteResults(
            eval_saved_model_path=model_location,
            output_path=output_path,
            display_only_data_location=data_location,
            example_weight_key=example_weight_key,
            add_metrics_callbacks=add_metrics_callbacks,
            slice_spec=slice_spec))
    # pylint: enable=no-value-for-parameter

  eval_result = load_eval_result(output_path=output_path)
  return eval_result


def multiple_model_analysis(model_locations, data_location,
                            **kwargs):
  """Run model analysis for multiple models on the same data set.

  Args:
    model_locations: A list of paths to the export eval saved model.
    data_location: The location of the data files.
    **kwargs: The args used for evaluation. See tfma.run_model_analysis() for
      details.

  Returns:
    A tfma.EvalResults containing all the evaluation results with the same order
    as model_locations.
  """
  results = []
  for m in model_locations:
    results.append(run_model_analysis(m, data_location, **kwargs))
  return api_types.EvalResults(results, constants.MODEL_CENTRIC_MODE)


def multiple_data_analysis(model_location, data_locations,
                           **kwargs):
  """Run model analysis for a single model on multiple data sets.

  Args:
    model_location: The location of the exported eval saved model.
    data_locations: A list of data set locations.
    **kwargs: The args used for evaluation. See tfma.run_model_analysis() for
      details.

  Returns:
    A tfma.EvalResults containing all the evaluation results with the same order
    as data_locations.
  """
  results = []
  for d in data_locations:
    results.append(run_model_analysis(model_location, d, **kwargs))
  return api_types.EvalResults(results, constants.DATA_CENTRIC_MODE)


def make_eval_results(results,
                      mode):
  """Run model analysis for a single model on multiple data sets.

  Args:
    results: A list of TFMA evaluation results.
    mode: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
    tfma.MODEL_CENTRIC_MODE are supported.

  Returns:
    An EvalResults containing all evaluation results. This can be used to
    construct a time series view.
  """
  return api_types.EvalResults(results, mode)


def load_eval_results(output_paths,
                      mode):
  """Run model analysis for a single model on multiple data sets.

  Args:
    output_paths: A list of output paths of completed tfma runs.
    mode: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
    tfma.MODEL_CENTRIC_MODE are supported.

  Returns:
    An EvalResults containing the evaluation results serialized at output_paths.
    This can be used to construct a time series view.
  """
  results = [load_eval_result(output_path) for output_path in output_paths]
  return make_eval_results(results, mode)
