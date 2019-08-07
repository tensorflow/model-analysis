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
# Standard __future__ imports
from __future__ import print_function

import os
import pickle
import tempfile

# Standard Imports

import apache_beam as beam
import six
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import version as tfma_version
from tensorflow_model_analysis.eval_saved_model import dofn
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
import tensorflow_model_analysis.post_export_metrics.metric_keys as metric_keys
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.validators import validator
from tensorflow_model_analysis.writers import metrics_and_plots_serialization
from tensorflow_model_analysis.writers import metrics_and_plots_writer
from tensorflow_model_analysis.writers import writer
from typing import Any, Dict, List, NamedTuple, Optional, Text, Tuple, Union

from google.protobuf import json_format

# File names for files written out to the result directory.
_METRICS_OUTPUT_FILE = 'metrics'
_PLOTS_OUTPUT_FILE = 'plots'
_EVAL_CONFIG_FILE = 'eval_config'

# Keys for the serialized final dictionary.
_VERSION_KEY = 'tfma_version'
_EVAL_CONFIG_KEY = 'eval_config'


def _assert_tensorflow_version():
  """Check that we're using a compatible TF version."""
  # Fail with a clear error in case we are not using a compatible TF version.
  major, minor, _ = tf.version.VERSION.split('.')
  if int(major) != 1 or int(minor) < 14:
    raise RuntimeError(
        'Tensorflow version >= 1.14, < 2 is required. Found (%s). Please '
        'install the latest 1.x version from '
        'https://github.com/tensorflow/tensorflow. ' % tf.version.VERSION)


class EvalConfig(
    NamedTuple(
        'EvalConfig',
        [
            # The location of the model used for this evaluation
            ('model_location', Text),
            # The location of the data used for this evaluation
            ('data_location', Text),
            # The corresponding slice spec
            ('slice_spec', Optional[List[slicer.SingleSliceSpec]]),
            # The example count metric key
            ('example_count_metric_key', Text),
            # The example weight metric key (or keys if multi-output model)
            ('example_weight_metric_key', Union[Text, Dict[Text, Text]]),
            # Set to true in order to calculate confidence intervals.
            ('compute_confidence_intervals', bool),
        ])):
  """Config used for extraction and evaluation."""

  def __new__(
      cls,
      model_location: Text,
      data_location: Optional[Text] = None,
      slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
      example_count_metric_key: Optional[Text] = None,
      example_weight_metric_key: Optional[Union[Text, Dict[Text, Text]]] = None,
      compute_confidence_intervals: Optional[bool] = False):
    return super(EvalConfig, cls).__new__(cls, model_location, data_location,
                                          slice_spec, example_count_metric_key,
                                          example_weight_metric_key,
                                          compute_confidence_intervals)


def _check_version(raw_final_dict: Dict[Text, Any], path: Text):
  version = raw_final_dict.get(_VERSION_KEY)
  if version is None:
    raise ValueError(
        'could not find TFMA version in raw deserialized dictionary for '
        'file at %s' % path)
  # We don't actually do any checking for now, since we don't have any
  # compatibility issues.


def _serialize_eval_config(eval_config: EvalConfig) -> bytes:
  final_dict = {}
  final_dict[_VERSION_KEY] = tfma_version.VERSION_STRING
  final_dict[_EVAL_CONFIG_KEY] = eval_config
  return pickle.dumps(final_dict)


def load_eval_config(output_path: Text) -> EvalConfig:
  serialized_record = six.next(
      tf.compat.v1.python_io.tf_record_iterator(
          os.path.join(output_path, _EVAL_CONFIG_FILE)))
  final_dict = pickle.loads(serialized_record)
  _check_version(final_dict, output_path)
  return final_dict[_EVAL_CONFIG_KEY]


EvalResult = NamedTuple(  # pylint: disable=invalid-name
    'EvalResult',
    [('slicing_metrics', List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]),
     ('plots', List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]),
     ('config', EvalConfig)])


class EvalResults(object):
  """Class for results from multiple model analysis run."""

  def __init__(self,
               results: List[EvalResult],
               mode: Text = constants.UNKNOWN_EVAL_MODE):
    supported_modes = [
        constants.DATA_CENTRIC_MODE,
        constants.MODEL_CENTRIC_MODE,
    ]
    if mode not in supported_modes:
      raise ValueError('Mode ' + mode + ' must be one of ' +
                       Text(supported_modes))

    self._results = results
    self._mode = mode

  def get_results(self) -> List[EvalResult]:
    return self._results

  def get_mode(self) -> Text:
    return self._mode


def make_eval_results(results: List[EvalResult], mode: Text) -> EvalResults:
  """Run model analysis for a single model on multiple data sets.

  Args:
    results: A list of TFMA evaluation results.
    mode: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
      tfma.MODEL_CENTRIC_MODE are supported.

  Returns:
    An EvalResults containing all evaluation results. This can be used to
    construct a time series view.
  """
  return EvalResults(results, mode)


def load_eval_results(output_paths: List[Text], mode: Text) -> EvalResults:
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


def load_eval_result(output_path: Text) -> EvalResult:
  """Creates an EvalResult object for use with the visualization functions."""
  metrics_proto_list = (
      metrics_and_plots_serialization.load_and_deserialize_metrics(
          path=os.path.join(output_path, _METRICS_OUTPUT_FILE)))
  plots_proto_list = (
      metrics_and_plots_serialization.load_and_deserialize_plots(
          path=os.path.join(output_path, _PLOTS_OUTPUT_FILE)))

  slicing_metrics = [(key, _convert_proto_map_to_dict(metrics_data))
                     for key, metrics_data in metrics_proto_list]
  plots = [(key, _convert_proto_map_to_dict(plot_data))
           for key, plot_data in plots_proto_list]

  eval_config = load_eval_config(output_path)
  return EvalResult(
      slicing_metrics=slicing_metrics, plots=plots, config=eval_config)


def default_eval_shared_model(
    eval_saved_model_path: Text,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    include_default_metrics: Optional[bool] = True,
    example_weight_key: Optional[Union[Text, Dict[Text, Text]]] = None,
    additional_fetches: Optional[List[Text]] = None,
    blacklist_feature_fetches: Optional[List[Text]] = None
) -> types.EvalSharedModel:
  """Returns default EvalSharedModel.

  Args:
    eval_saved_model_path: Path to EvalSavedModel.
    add_metrics_callbacks: Optional list of callbacks for adding additional
      metrics to the graph (see EvalSharedModel for more information on how to
      configure additional metrics). Metrics for example count and example
      weights will be added automatically.
    include_default_metrics: True to include the default metrics that are part
      of the saved model graph during evaluation.
    example_weight_key: Example weight key (single-output model) or dict of
      example weight keys (multi-output model) keyed by output name.
    additional_fetches: Prefixes of additional tensors stored in
      signature_def.inputs that should be fetched at prediction time. The
      "features" and "labels" tensors are handled automatically and should not
      be included.
    blacklist_feature_fetches: List of tensor names in the features dictionary
      which should be excluded from the fetches request. This is useful in
      scenarios where features are large (e.g. images) and can lead to excessive
      memory use if stored.
  """
  # Always compute example weight and example count.
  # PyType doesn't know about the magic exports we do in post_export_metrics.
  # Additionally, the lines seem to get reordered in compilation, so we can't
  # just put the disable-attr on the add_metrics_callbacks lines.
  # pytype: disable=module-attr
  if not add_metrics_callbacks:
    add_metrics_callbacks = []
  example_count_callback = post_export_metrics.example_count()
  add_metrics_callbacks.append(example_count_callback)
  if example_weight_key:
    if isinstance(example_weight_key, dict):
      for output_name, key in example_weight_key.items():
        example_weight_callback = post_export_metrics.example_weight(
            key, metric_tag=output_name)
        add_metrics_callbacks.append(example_weight_callback)
    else:
      example_weight_callback = post_export_metrics.example_weight(
          example_weight_key)
      add_metrics_callbacks.append(example_weight_callback)
  # pytype: enable=module-attr

  return types.EvalSharedModel(
      model_path=eval_saved_model_path,
      add_metrics_callbacks=add_metrics_callbacks,
      include_default_metrics=include_default_metrics,
      example_weight_key=example_weight_key,
      additional_fetches=additional_fetches,
      construct_fn=dofn.make_construct_fn(
          eval_saved_model_path,
          add_metrics_callbacks,
          include_default_metrics,
          additional_fetches=additional_fetches,
          blacklist_feature_fetches=blacklist_feature_fetches))


def default_extractors(  # pylint: disable=invalid-name
    eval_shared_model: types.EvalSharedModel,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    desired_batch_size: Optional[int] = None,
    materialize: Optional[bool] = True) -> List[extractor.Extractor]:
  """Returns the default extractors for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    desired_batch_size: Optional batch size for batching in Aggregate.
    materialize: True to have extractors create materialized output.
  """
  return [
      predict_extractor.PredictExtractor(
          eval_shared_model, desired_batch_size, materialize=materialize),
      slice_key_extractor.SliceKeyExtractor(
          slice_spec, materialize=materialize)
  ]


def default_evaluators(  # pylint: disable=invalid-name
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1) -> List[evaluator.Evaluator]:
  """Returns the default evaluators for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    compute_confidence_intervals: Whether or not to compute confidence
      intervals.
    k_anonymization_count: If the number of examples in a specific slice is less
      than k_anonymization_count, then an error will be returned for that slice.
      This will be useful to ensure privacy by not displaying the aggregated
      data for smaller number of examples.
  """
  return [
      metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(
          eval_shared_model,
          desired_batch_size,
          compute_confidence_intervals=compute_confidence_intervals,
          k_anonymization_count=k_anonymization_count)
  ]


def default_writers(eval_shared_model: types.EvalSharedModel,
                    output_path: Text) -> List[writer.Writer]:  # pylint: disable=invalid-name
  """Returns the default writers for use in WriteResults.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    output_path: Path to store results files under.
  """
  output_paths = {
      constants.METRICS_KEY: os.path.join(output_path, _METRICS_OUTPUT_FILE),
      constants.PLOTS_KEY: os.path.join(output_path, _PLOTS_OUTPUT_FILE)
  }
  return [
      metrics_and_plots_writer.MetricsAndPlotsWriter(
          eval_shared_model=eval_shared_model, output_paths=output_paths)
  ]


# The input type is a MessageMap where the keys are strings and the values are
# some protocol buffer field. Note that MessageMap is not a protobuf message,
# none of the exising utility methods work on it. We must iterate over its
# values and call the utility function individually.
def _convert_proto_map_to_dict(proto_map: Dict[Text, Any]) -> Dict[Text, Any]:
  """Converts a metric map (metrics in MetricsForSlice protobuf) into a dict.

  Args:
    proto_map: A protocol buffer MessageMap that has behaviors like dict. The
      keys are strings while the values are protocol buffers. However, it is not
      a protobuf message and cannot be passed into json_format.MessageToDict
      directly. Instead, we must iterate over its values.

  Returns:
    A dict representing the proto_map. For example:
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

    The output of _convert_proto_map_to_dict(myProto.metrics) would be

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
  return {k: json_format.MessageToDict(proto_map[k]) for k in proto_map}


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(types.Extracts)
def InputsToExtracts(  # pylint: disable=invalid-name
    inputs: beam.pvalue.PCollection):
  """Converts serialized inputs (e.g. examples) to Extracts."""
  return inputs | beam.Map(lambda x: {constants.INPUT_KEY: x})


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(evaluator.Evaluation)
def ExtractAndEvaluate(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection, extractors: List[extractor.Extractor],
    evaluators: List[evaluator.Evaluator]):
  """Performs Extractions and Evaluations in provided order."""
  # evaluation[k] = list of values for k
  evaluation = {}

  def update(evaluation: Dict[Text, Any], new_evaluation: Dict[Text, Any]):
    for k, v in new_evaluation.items():
      if k not in evaluation:
        evaluation[k] = []
      evaluation[k].append(v)
    return evaluation

  # Run evaluators that run before extraction (i.e. that only require
  # the incoming input extract added by ReadInputs)
  for v in evaluators:
    if not v.run_after:
      update(evaluation, extracts | v.stage_name >> v.ptransform)
  for x in extractors:
    extracts = (extracts | x.stage_name >> x.ptransform)
    for v in evaluators:
      if v.run_after == x.stage_name:
        update(evaluation, extracts | v.stage_name >> v.ptransform)
  for v in evaluators:
    if v.run_after == extractor.LAST_EXTRACTOR_STAGE_NAME:
      update(evaluation, extracts | v.stage_name >> v.ptransform)

  # Merge multi-valued keys if necessary.
  result = {}
  for k, v in evaluation.items():
    if len(v) == 1:
      result[k] = v[0]
      continue

    # Note that we assume that if a key is multivalued, its values are
    # dictionaries with disjoint keys. The combined value will simply be the
    # disjoint union of all the dictionaries.
    result[k] = (
        v
        | 'FlattenEvaluationOutput(%s)' % k >> beam.Flatten()
        | 'CombineEvaluationOutput(%s)' % k >> beam.CombinePerKey(
            _CombineEvaluationDictionariesFn()))

  return result


class _CombineEvaluationDictionariesFn(beam.CombineFn):
  """CombineFn to combine dictionaries generated by different evaluators."""

  def create_accumulator(self) -> Dict[Text, Any]:
    return {}

  def _merge(self, accumulator: Dict[Text, Any],
             output_dict: Dict[Text, Any]) -> None:
    intersection = set(accumulator) & set(output_dict)
    if intersection:
      raise ValueError(
          'Dictionaries generated by different evaluators should have '
          'different keys, but keys %s appeared in the output of multiple '
          'evaluators' % intersection)
    accumulator.update(output_dict)

  def add_input(self, accumulator: Dict[Text, Any],
                output_dict: Dict[Text, Any]) -> Dict[Text, Any]:
    if not isinstance(output_dict, dict):
      raise TypeError(
          'for outputs written to by multiple evaluators, the outputs must all '
          'be dictionaries, but got output of type %s, value %s' %
          (type(output_dict), str(output_dict)))
    self._merge(accumulator, output_dict)
    return accumulator

  def merge_accumulators(self, accumulators: List[Dict[Text, Any]]
                        ) -> Dict[Text, Any]:
    result = self.create_accumulator()
    for acc in accumulators:
      self._merge(result, acc)
    return result

  def extract_output(self, accumulator: Dict[Text, Any]) -> Dict[Text, Any]:
    return accumulator


@beam.ptransform_fn
@beam.typehints.with_input_types(
    Union[evaluator.Evaluation, validator.Validation])
@beam.typehints.with_output_types(beam.pvalue.PDone)
def WriteResults(  # pylint: disable=invalid-name
    evaluation_or_validation: Union[evaluator.Evaluation, validator.Validation],
    writers: List[writer.Writer]):
  """Writes Evaluation or Validation results using given writers.

  Args:
    evaluation_or_validation: Evaluation or Validation output.
    writers: Writes to use for writing out output.

  Raises:
    ValueError: If Evaluation or Validation is empty.

  Returns:
    beam.pvalue.PDone.
  """
  if not evaluation_or_validation:
    raise ValueError('Evaluations and Validations cannot be empty')
  for w in writers:
    _ = evaluation_or_validation | w.stage_name >> w.ptransform
  return beam.pvalue.PDone(list(evaluation_or_validation.values())[0].pipeline)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def WriteEvalConfig(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, eval_config: EvalConfig, output_path: Text):
  """Writes EvalConfig to file.

  Args:
    pipeline: Beam pipeline.
    eval_config: EvalConfig.
    output_path: Path to store output under.

  Returns:
    beam.pvalue.PDone.
  """
  return (
      pipeline
      | 'CreateEvalConfig' >> beam.Create([_serialize_eval_config(eval_config)])
      | 'WriteEvalConfig' >> beam.io.WriteToTFRecord(
          os.path.join(output_path, _EVAL_CONFIG_FILE), shard_name_template=''))


@beam.ptransform_fn
@beam.typehints.with_output_types(beam.pvalue.PDone)
def ExtractEvaluateAndWriteResults(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    output_path: Text,
    display_only_data_location: Optional[Text] = None,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    desired_batch_size: Optional[int] = None,
    extractors: Optional[List[extractor.Extractor]] = None,
    evaluators: Optional[List[evaluator.Evaluator]] = None,
    writers: Optional[List[writer.Writer]] = None,
    write_config: Optional[bool] = True,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1) -> beam.pvalue.PDone:
  """PTransform for performing extraction, evaluation, and writing results.

  Users who want to construct their own Beam pipelines instead of using the
  lightweight run_model_analysis functions should use this PTransform.

  Example usage:
    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[...])
    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
           | 'ExtractEvaluateAndWriteResults' >>
           tfma.ExtractEvaluateAndWriteResults(
               eval_shared_model=eval_shared_model,
               output_path=output_path,
               display_only_data_location=data_location,
               slice_spec=slice_spec,
               ...))
    result = tfma.load_eval_result(output_path=output_path)
    tfma.view.render_slicing_metrics(result)

  Note that the exact serialization format is an internal implementation detail
  and subject to change. Users should only use the TFMA functions to write and
  read the results.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    output_path: Path to output metrics and plots results.
    display_only_data_location: Optional path indicating where the examples were
      read from. This is used only for display purposes - data will not actually
      be read from this path.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.
    extractors: Optional list of Extractors to apply to Extracts. Typically
      these will be added by calling the default_extractors function. If no
      extractors are provided, default_extractors (non-materialized) will be
      used.
    evaluators: Optional list of Evaluators for evaluating Extracts. Typically
      these will be added by calling the default_evaluators function. If no
      evaluators are provided, default_evaluators will be used.
    writers: Optional list of Writers for writing Evaluation output. Typically
      these will be added by calling the default_writers function. If no writers
      are provided, default_writers will be used.
    write_config: True to write the config along with the results.
    compute_confidence_intervals: If true, compute confidence intervals.
    k_anonymization_count: If the number of examples in a specific slice is less
      than k_anonymization_count, then an error will be returned for that slice.
      This will be useful to ensure privacy by not displaying the aggregated
      data for smaller number of examples.

  Raises:
    ValueError: If matching Extractor not found for an Evaluator.

  Returns:
    PDone.
  """
  if not extractors:
    extractors = default_extractors(
        eval_shared_model=eval_shared_model,
        slice_spec=slice_spec,
        desired_batch_size=desired_batch_size,
        materialize=False)

  if not evaluators:
    evaluators = default_evaluators(
        eval_shared_model=eval_shared_model,
        desired_batch_size=desired_batch_size,
        compute_confidence_intervals=compute_confidence_intervals,
        k_anonymization_count=k_anonymization_count)

  for v in evaluators:
    evaluator.verify_evaluator(v, extractors)

  if not writers:
    writers = default_writers(
        eval_shared_model=eval_shared_model, output_path=output_path)

  data_location = '<user provided PCollection>'
  if display_only_data_location is not None:
    data_location = display_only_data_location

  example_weight_metric_key = metric_keys.EXAMPLE_COUNT
  if eval_shared_model.example_weight_key:
    if isinstance(eval_shared_model.example_weight_key, dict):
      example_weight_metric_key = {}
      for output_name in eval_shared_model.example_weight_key:
        example_weight_metric_key[output_name] = metric_keys.tagged_key(
            metric_keys.EXAMPLE_WEIGHT, output_name)
    else:
      example_weight_metric_key = metric_keys.EXAMPLE_WEIGHT

  eval_config = EvalConfig(
      model_location=eval_shared_model.model_path,
      data_location=data_location,
      slice_spec=slice_spec,
      example_count_metric_key=metric_keys.EXAMPLE_COUNT,
      example_weight_metric_key=example_weight_metric_key,
      compute_confidence_intervals=compute_confidence_intervals)

  # pylint: disable=no-value-for-parameter
  _ = (
      examples
      | 'InputsToExtracts' >> InputsToExtracts()
      | 'ExtractAndEvaluate' >> ExtractAndEvaluate(
          extractors=extractors, evaluators=evaluators)
      | 'WriteResults' >> WriteResults(writers=writers))

  if write_config:
    _ = examples.pipeline | WriteEvalConfig(eval_config, output_path)
  # pylint: enable=no-value-for-parameter

  return beam.pvalue.PDone(examples.pipeline)


def run_model_analysis(
    eval_shared_model: types.EvalSharedModel,
    data_location: Text,
    file_format: Text = 'tfrecords',
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    output_path: Optional[Text] = None,
    extractors: Optional[List[extractor.Extractor]] = None,
    evaluators: Optional[List[evaluator.Evaluator]] = None,
    writers: Optional[List[writer.Writer]] = None,
    write_config: Optional[bool] = True,
    desired_batch_size: Optional[int] = None,
    pipeline_options: Optional[Any] = None,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1,
) -> EvalResult:
  """Runs TensorFlow model analysis.

  It runs a Beam pipeline to compute the slicing metrics exported in TensorFlow
  Eval SavedModel and returns the results.

  This is a simplified API for users who want to quickly get something running
  locally. Users who wish to create their own Beam pipelines can use the
  Evaluate PTransform instead.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    data_location: The location of the data files.
    file_format: The file format of the data, can be either 'text' or
      'tfrecords' for now. By default, 'tfrecords' will be used.
    slice_spec: A list of tfma.slicer.SingleSliceSpec. Each spec represents a
      way to slice the data. If None, defaults to the overall slice.
      Example usages:
      # TODO(xinzha): add more use cases once they are supported in frontend.
      - tfma.SingleSiceSpec(): no slice, metrics are computed on overall data.
      - tfma.SingleSiceSpec(columns=['country']): slice based on features in
        column "country". We might get metrics for slice "country:us",
        "country:jp", and etc in results.
      - tfma.SingleSiceSpec(features=[('country', 'us')]): metrics are computed
        on slice "country:us".
    output_path: The directory to output metrics and results to. If None, we use
      a temporary directory.
    extractors: Optional list of Extractors to apply to Extracts. Typically
      these will be added by calling the default_extractors function. If no
      extractors are provided, default_extractors (non-materialized) will be
      used.
    evaluators: Optional list of Evaluators for evaluating Extracts. Typically
      these will be added by calling the default_evaluators function. If no
      evaluators are provided, default_evaluators will be used.
    writers: Optional list of Writers for writing Evaluation output. Typically
      these will be added by calling the default_writers function. If no writers
      are provided, default_writers will be used.
    write_config: True to write the config along with the results.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.
    pipeline_options: Optional arguments to run the Pipeline, for instance
      whether to run directly.
    compute_confidence_intervals: If true, compute confidence intervals.
    k_anonymization_count: If the number of examples in a specific slice is less
      than k_anonymization_count, then an error will be returned for that slice.
      This will be useful to ensure privacy by not displaying the aggregated
      data for smaller number of examples.

  Returns:
    An EvalResult that can be used with the TFMA visualization functions.

  Raises:
    ValueError: If the file_format is unknown to us.
  """
  _assert_tensorflow_version()
  # Get working_dir ready.
  if output_path is None:
    output_path = tempfile.mkdtemp()
  if not tf.io.gfile.exists(output_path):
    tf.io.gfile.makedirs(output_path)

  with beam.Pipeline(options=pipeline_options) as p:
    if file_format == 'tfrecords':
      data = p | 'ReadFromTFRecord' >> beam.io.ReadFromTFRecord(
          file_pattern=data_location,
          compression_type=beam.io.filesystem.CompressionTypes.AUTO)
    elif file_format == 'text':
      data = p | 'ReadFromText' >> beam.io.textio.ReadFromText(data_location)
    else:
      raise ValueError('unknown file_format: %s' % file_format)

    # pylint: disable=no-value-for-parameter
    _ = (
        data
        | 'ExtractEvaluateAndWriteResults' >> ExtractEvaluateAndWriteResults(
            eval_shared_model=eval_shared_model,
            output_path=output_path,
            display_only_data_location=data_location,
            slice_spec=slice_spec,
            extractors=extractors,
            evaluators=evaluators,
            writers=writers,
            write_config=write_config,
            desired_batch_size=desired_batch_size,
            compute_confidence_intervals=compute_confidence_intervals,
            k_anonymization_count=k_anonymization_count))
    # pylint: enable=no-value-for-parameter

  eval_result = load_eval_result(output_path=output_path)
  return eval_result


def multiple_model_analysis(model_locations: List[Text], data_location: Text,
                            **kwargs) -> EvalResults:
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
    results.append(
        run_model_analysis(
            default_eval_shared_model(m), data_location, **kwargs))
  return EvalResults(results, constants.MODEL_CENTRIC_MODE)


def multiple_data_analysis(model_location: Text, data_locations: List[Text],
                           **kwargs) -> EvalResults:
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
    results.append(
        run_model_analysis(
            default_eval_shared_model(model_location), d, **kwargs))
  return EvalResults(results, constants.DATA_CENTRIC_MODE)
