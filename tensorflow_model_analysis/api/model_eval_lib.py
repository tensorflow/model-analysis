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
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis import version as tfma_version
from tensorflow_model_analysis.eval_saved_model import constants as eval_constants
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.validators import validator
from tensorflow_model_analysis.writers import metrics_and_plots_serialization
from tensorflow_model_analysis.writers import metrics_and_plots_writer
from tensorflow_model_analysis.writers import writer
from typing import Any, Dict, List, NamedTuple, Optional, Text, Tuple, Union

from google.protobuf import json_format

_EVAL_CONFIG_FILE = 'eval_config.json'


def _assert_tensorflow_version():
  """Check that we're using a compatible TF version."""
  # Fail with a clear error in case we are not using a compatible TF version.
  major, minor, _ = tf.version.VERSION.split('.')
  if (int(major) not in (1, 2)) or (int(major == 1 and int(minor) < 15)):
    raise RuntimeError(
        'Tensorflow version >= 1.15, < 3 is required. Found (%s). Please '
        'install the latest 1.x or 2.x version from '
        'https://github.com/tensorflow/tensorflow. ' % tf.version.VERSION)
  if int(major) == 2:
    tf.compat.v1.logging.warning(
        'Tensorflow version (%s) found. Note that TFMA support for TF 2.0 '
        'is currently in beta' % tf.version.VERSION)


def _check_version(version: Text, path: Text):
  if not version:
    raise ValueError(
        'could not find TFMA version in raw deserialized dictionary for '
        'file at %s' % path)
  # We don't actually do any checking for now, since we don't have any
  # compatibility issues.


def _serialize_eval_config(eval_config: config.EvalConfig) -> Text:
  return json_format.MessageToJson(
      config_pb2.EvalConfigAndVersion(
          eval_config=eval_config, version=tfma_version.VERSION_STRING))


def load_eval_config(output_path: Text) -> config.EvalConfig:
  """Loads eval config."""
  path = os.path.join(output_path, _EVAL_CONFIG_FILE)
  if tf.io.gfile.exists(path):
    with tf.io.gfile.GFile(path, 'r') as f:
      pb = json_format.Parse(f.read(), config_pb2.EvalConfigAndVersion())
      _check_version(pb.version, output_path)
      return pb.eval_config
  else:
    # Legacy suppport (to be removed in future).
    # The previous version did not include file extension.
    path = os.path.splitext(path)[0]
    serialized_record = six.next(
        tf.compat.v1.python_io.tf_record_iterator(path))
    final_dict = pickle.loads(serialized_record)
    _check_version(final_dict, output_path)
    old_config = final_dict['eval_config']
    slicing_specs = None
    if old_config.slice_spec:
      slicing_specs = [s.to_proto() for s in old_config.slice_spec]
    options = config.Options()
    options.compute_confidence_intervals.value = (
        old_config.compute_confidence_intervals)
    options.k_anonymization_count.value = old_config.k_anonymization_count
    return config.EvalConfig(
        input_data_specs=[
            config.InputDataSpec(location=old_config.data_location)
        ],
        model_specs=[config.ModelSpec(location=old_config.model_location)],
        output_data_specs=[config.OutputDataSpec(default_location=output_path)],
        slicing_specs=slicing_specs,
        options=options)


# The field slicing_metrics is a nested dictionaries representing metrics for
# different configuration as defined by MetricKey in metrics_for_slice.proto.
# The levels corresponds to output name, class id, metric name and metric value
# in this order. Note MetricValue uses oneof so metric values will always
# contain only a single key representing the type in the oneof and the actual
# metric value is in the value.
EvalResult = NamedTuple(  # pylint: disable=invalid-name
    'EvalResult',
    [('slicing_metrics',
      List[Tuple[slicer.SliceKeyType,
                 Dict[Text, Dict[Text, Dict[Text, Dict[Text, Dict[Text,
                                                                  Any]]]]]]]),
     ('plots', List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]),
     ('config', config.EvalConfig)])


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


def load_eval_results(output_paths: List[Text],
                      mode: Text,
                      model_name: Optional[Text] = None) -> EvalResults:
  """Run model analysis for a single model on multiple data sets.

  Args:
    output_paths: A list of output paths of completed tfma runs.
    mode: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
      tfma.MODEL_CENTRIC_MODE are supported.
    model_name: The name of the model if multiple models are evaluated together.

  Returns:
    An EvalResults containing the evaluation results serialized at output_paths.
    This can be used to construct a time series view.
  """
  results = [
      load_eval_result(output_path, model_name=model_name)
      for output_path in output_paths
  ]
  return make_eval_results(results, mode)


def load_eval_result(output_path: Text,
                     model_name: Optional[Text] = None) -> EvalResult:
  """Creates an EvalResult object for use with the visualization functions."""
  eval_config = load_eval_config(output_path)
  output_spec = _get_output_data_spec(eval_config, model_name)
  metrics_proto_list = (
      metrics_and_plots_serialization.load_and_deserialize_metrics(
          path=output_filename(output_spec, constants.METRICS_KEY),
          model_name=model_name))
  plots_proto_list = (
      metrics_and_plots_serialization.load_and_deserialize_plots(
          path=output_filename(output_spec, constants.PLOTS_KEY)))

  return EvalResult(
      slicing_metrics=metrics_proto_list,
      plots=plots_proto_list,
      config=eval_config)


def _get_output_data_spec(eval_config: config.EvalConfig,
                          model_name: Text) -> Optional[config.OutputDataSpec]:
  """Returns output data spec with given model name or default."""
  for spec in eval_config.output_data_specs:
    if spec.model_name == model_name:
      return spec
  for spec in eval_config.output_data_specs:
    if not spec.model_name:
      return spec
  return None


def output_filename(spec: config.OutputDataSpec, key: Text) -> Text:
  """Returns output filename for given key."""
  location = spec.default_location
  if spec.custom_locations and key in spec.custom_locations:
    location = spec.custom_locations[key]
  return os.path.join(location, key)


def default_eval_shared_model(
    eval_saved_model_path: Text,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    include_default_metrics: Optional[bool] = True,
    example_weight_key: Optional[Union[Text, Dict[Text, Text]]] = None,
    additional_fetches: Optional[List[Text]] = None,
    blacklist_feature_fetches: Optional[List[Text]] = None,
    tags: Optional[List[Text]] = None) -> types.EvalSharedModel:
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
    tags: Model tags (e.g. 'serve' for serving or 'eval' for EvalSavedModel).
  """
  if tags is None:
    tags = [eval_constants.EVAL_TAG]

  # Backwards compatibility for previous EvalSavedModel implementation.
  if tags == [eval_constants.EVAL_TAG]:
    # PyType doesn't know about the magic exports we do in post_export_metrics.
    # Additionally, the lines seem to get reordered in compilation, so we can't
    # just put the disable-attr on the add_metrics_callbacks lines.
    # pytype: disable=module-attr
    if not add_metrics_callbacks:
      add_metrics_callbacks = []
    # Always compute example weight and example count.
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
      model_loader=types.ModelLoader(
          tags=tags,
          construct_fn=model_util.model_construct_fn(
              eval_saved_model_path=eval_saved_model_path,
              add_metrics_callbacks=add_metrics_callbacks,
              include_default_metrics=include_default_metrics,
              additional_fetches=additional_fetches,
              blacklist_feature_fetches=blacklist_feature_fetches,
              tags=tags)))


def default_extractors(  # pylint: disable=invalid-name
    eval_shared_model: Optional[types.EvalSharedModel] = None,
    eval_shared_models: Optional[List[types.EvalSharedModel]] = None,
    eval_config: config.EvalConfig = None,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    desired_batch_size: Optional[int] = None,
    materialize: Optional[bool] = True) -> List[extractor.Extractor]:
  """Returns the default extractors for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Shared model (single-model evaluation).
    eval_shared_models: Shared models (multi-model evaluation).
    eval_config: Eval config.
    slice_spec: Deprecated (use EvalConfig).
    desired_batch_size: Deprecated (use EvalConfig).
    materialize: True to have extractors create materialized output.
  """
  # TODO(b/141016373): Add support for multiple models.
  if eval_config is not None:
    slice_spec = [
        slicer.SingleSliceSpec(spec=spec) for spec in eval_config.slicing_specs
    ]
    if eval_config.options.HasField('desired_batch_size'):
      desired_batch_size = eval_config.options.desired_batch_size.value
  if eval_shared_model is not None:
    eval_shared_models = [eval_shared_model]
  if (not eval_shared_models[0].model_loader.tags or
      eval_shared_models[0].model_loader.tags == [eval_constants.EVAL_TAG]):
    # Backwards compatibility for previous EvalSavedModel implementation.
    return [
        predict_extractor.PredictExtractor(
            eval_shared_models[0], desired_batch_size, materialize=materialize),
        slice_key_extractor.SliceKeyExtractor(
            slice_spec, materialize=materialize)
    ]
  else:
    raise NotImplementedError('keras and serving models not implemented yet.')


def default_evaluators(  # pylint: disable=invalid-name
    eval_shared_model: Optional[types.EvalSharedModel] = None,
    eval_shared_models: Optional[List[types.EvalSharedModel]] = None,
    eval_config: config.EvalConfig = None,
    desired_batch_size: Optional[int] = None,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1) -> List[evaluator.Evaluator]:
  """Returns the default evaluators for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Shared model (single-model evaluation).
    eval_shared_models: Shared models (multi-model evaluation).
    eval_config: Eval config.
    desired_batch_size: Deprecated (use eval_config).
    compute_confidence_intervals: Deprecated (use eval_config).
    k_anonymization_count: Deprecated (use eval_config).
  """
  # TODO(b/141016373): Add support for multiple models.
  if eval_shared_model is not None:
    eval_shared_models = [eval_shared_model]
  if not eval_config or not eval_config.metrics_specs:
    # Backwards compatibility for previous EvalSavedModel implementation.
    if eval_config is not None:
      if eval_config.options.HasField('desired_batch_size'):
        desired_batch_size = eval_config.options.desired_batch_size.value
      if eval_config.options.HasField('compute_confidence_intervals'):
        compute_confidence_intervals = (
            eval_config.options.compute_confidence_intervals.value)
      if eval_config.options.HasField('k_anonymization_count'):
        k_anonymization_count = eval_config.options.k_anonymization_count.value
    return [
        metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(
            eval_shared_models[0],
            desired_batch_size,
            compute_confidence_intervals=compute_confidence_intervals,
            k_anonymization_count=k_anonymization_count)
    ]
  else:
    raise NotImplementedError('metrics_specs not implemented yet.')


def default_writers(
    eval_shared_model: Optional[types.EvalSharedModel] = None,
    eval_shared_models: Optional[List[types.EvalSharedModel]] = None,
    output_path: Optional[Text] = None,
    eval_config: config.EvalConfig = None,
) -> List[writer.Writer]:  # pylint: disable=invalid-name
  """Returns the default writers for use in WriteResults.

  Args:
    eval_shared_model: Shared model (single-model evaluation).
    eval_shared_models: Shared models (multi-model evaluation).
    output_path: Deprecated (use EvalConfig).
    eval_config: Eval config.
  """
  # TODO(b/141016373): Add support for multiple models.
  if eval_config is not None:
    output_spec = eval_config.output_data_specs[0]
  elif output_path is not None:
    output_spec = config.OutputDataSpec(default_location=output_path)
  if eval_shared_model is not None:
    eval_shared_models = [eval_shared_model]
  output_paths = {
      constants.METRICS_KEY:
          output_filename(output_spec, constants.METRICS_KEY),
      constants.PLOTS_KEY:
          output_filename(output_spec, constants.PLOTS_KEY)
  }
  return [
      metrics_and_plots_writer.MetricsAndPlotsWriter(
          eval_shared_model=eval_shared_models[0], output_paths=output_paths)
  ]


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

  def merge_accumulators(
      self, accumulators: List[Dict[Text, Any]]) -> Dict[Text, Any]:
    result = self.create_accumulator()
    for acc in accumulators:
      self._merge(result, acc)
    return result

  def extract_output(self, accumulator: Dict[Text, Any]) -> Dict[Text, Any]:
    return accumulator


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[evaluator.Evaluation,
                                       validator.Validation])
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
    pipeline: beam.Pipeline, eval_config: config.EvalConfig):
  """Writes EvalConfig to file.

  Args:
    pipeline: Beam pipeline.
    eval_config: EvalConfig.

  Returns:
    beam.pvalue.PDone.
  """
  # TODO(b/141016373): Add support for multiple models.
  output_path = output_filename(eval_config.output_data_specs[0],
                                _EVAL_CONFIG_FILE)
  return (
      pipeline
      | 'CreateEvalConfig' >> beam.Create([_serialize_eval_config(eval_config)])
      | 'WriteEvalConfig' >> beam.io.WriteToText(
          output_path, shard_name_template=''))


@beam.ptransform_fn
@beam.typehints.with_output_types(beam.pvalue.PDone)
def ExtractEvaluateAndWriteResults(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    eval_shared_model: Optional[types.EvalSharedModel] = None,
    eval_shared_models: Optional[List[types.EvalSharedModel]] = None,
    eval_config: config.EvalConfig = None,
    extractors: Optional[List[extractor.Extractor]] = None,
    evaluators: Optional[List[evaluator.Evaluator]] = None,
    writers: Optional[List[writer.Writer]] = None,
    output_path: Optional[Text] = None,
    display_only_data_location: Optional[Text] = None,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    desired_batch_size: Optional[int] = None,
    write_config: Optional[bool] = True,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1) -> beam.pvalue.PDone:
  """PTransform for performing extraction, evaluation, and writing results.

  Users who want to construct their own Beam pipelines instead of using the
  lightweight run_model_analysis functions should use this PTransform.

  Example usage:
    eval_config = tfma.EvalConfig(
        input_data_specs=[tfma.InputDataSpec(location=data_location)],
        model_specs=[tfma.ModelSpec(location=model_location)],
        output_data_specs=[tfma.OutputDataSpec(default_location=output_path)],
        slicing_specs=[...],
        metrics_specs=[...])
    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[...])
    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
           | 'ExtractEvaluateAndWriteResults' >>
           tfma.ExtractEvaluateAndWriteResults(
               eval_config=eval_config,
               eval_shared_models=[eval_shared_model],
               ...))
    result = tfma.load_eval_result(output_path=output_path)
    tfma.view.render_slicing_metrics(result)

  Note that the exact serialization format is an internal implementation detail
  and subject to change. Users should only use the TFMA functions to write and
  read the results.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_shared_model: Shared model (single-model evaluation).
    eval_shared_models: Shared models (multi-model evaluation).
    eval_config: Eval config.
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
    output_path: Deprecated (use EvalConfig).
    display_only_data_location: Deprecated (use EvalConfig).
    slice_spec: Deprecated (use EvalConfig).
    desired_batch_size: Deprecated (use EvalConfig).
    write_config: Deprecated (use EvalConfig).
    compute_confidence_intervals: Deprecated (use EvalConfig).
    k_anonymization_count: Deprecated (use EvalConfig).

  Raises:
    ValueError: If matching Extractor not found for an Evaluator.

  Returns:
    PDone.
  """
  if eval_shared_model is not None:
    eval_shared_models = [eval_shared_model]

  if eval_config is None:
    data_location = '<user provided PCollection>'
    if display_only_data_location is not None:
      data_location = display_only_data_location
    disabled_outputs = None
    if not write_config:
      disabled_outputs = [_EVAL_CONFIG_FILE]
    model_specs = []
    for m in eval_shared_models:
      example_weight_key = m.example_weight_key
      example_weight_keys = {}
      if example_weight_key and isinstance(example_weight_key, dict):
        example_weight_keys = example_weight_key
        example_weight_key = ''
      model_specs.append(
          config.ModelSpec(
              location=m.model_path,
              example_weight_key=example_weight_key,
              example_weight_keys=example_weight_keys))
    slicing_specs = None
    if slice_spec:
      slicing_specs = [s.to_proto() for s in slice_spec]
    options = config.Options()
    options.compute_confidence_intervals.value = compute_confidence_intervals
    options.k_anonymization_count.value = k_anonymization_count
    if desired_batch_size:
      options.desired_batch_size.value = desired_batch_size
    eval_config = config.EvalConfig(
        input_data_specs=[config.InputDataSpec(location=data_location)],
        model_specs=model_specs,
        output_data_specs=[
            config.OutputDataSpec(
                default_location=output_path, disabled_outputs=disabled_outputs)
        ],
        slicing_specs=slicing_specs,
        options=options)

  if not extractors:
    extractors = default_extractors(
        eval_config=eval_config,
        eval_shared_models=eval_shared_models,
        materialize=False)

  if not evaluators:
    evaluators = default_evaluators(
        eval_config=eval_config, eval_shared_models=eval_shared_models)

  for v in evaluators:
    evaluator.verify_evaluator(v, extractors)

  if not writers:
    writers = default_writers(
        eval_config=eval_config, eval_shared_models=eval_shared_models)

  # pylint: disable=no-value-for-parameter
  _ = (
      examples
      | 'InputsToExtracts' >> InputsToExtracts()
      | 'ExtractAndEvaluate' >> ExtractAndEvaluate(
          extractors=extractors, evaluators=evaluators)
      | 'WriteResults' >> WriteResults(writers=writers))

  # TODO(b/141016373): Add support for multiple models.
  if _EVAL_CONFIG_FILE not in eval_config.output_data_specs[0].disabled_outputs:
    _ = examples.pipeline | WriteEvalConfig(eval_config)
  # pylint: enable=no-value-for-parameter

  return beam.pvalue.PDone(examples.pipeline)


def run_model_analysis(
    eval_shared_model: Optional[types.EvalSharedModel] = None,
    eval_shared_models: Optional[List[types.EvalSharedModel]] = None,
    eval_config: config.EvalConfig = None,
    extractors: Optional[List[extractor.Extractor]] = None,
    evaluators: Optional[List[evaluator.Evaluator]] = None,
    writers: Optional[List[writer.Writer]] = None,
    pipeline_options: Optional[Any] = None,
    data_location: Optional[Text] = None,
    file_format: Optional[Text] = 'tfrecords',
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    output_path: Optional[Text] = None,
    write_config: Optional[bool] = True,
    desired_batch_size: Optional[int] = None,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1) -> EvalResult:
  """Runs TensorFlow model analysis.

  It runs a Beam pipeline to compute the slicing metrics exported in TensorFlow
  Eval SavedModel and returns the results.

  This is a simplified API for users who want to quickly get something running
  locally. Users who wish to create their own Beam pipelines can use the
  Evaluate PTransform instead.

  Args:
    eval_shared_model: Shared model (single-model evaluation).
    eval_shared_models: Shared models (multi-model evaluation).
    eval_config: Eval config.
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
    pipeline_options: Optional arguments to run the Pipeline, for instance
      whether to run directly.
    data_location: Deprecated (use EvalConfig).
    file_format: Deprecated (use EvalConfig).
    slice_spec: Deprecated (use EvalConfig).
    output_path: Deprecated (use EvalConfig).
    write_config: Deprecated (use EvalConfig).
    desired_batch_size: Deprecated (use EvalConfig).
    compute_confidence_intervals: Deprecated (use EvalConfig).
    k_anonymization_count: Deprecated (use EvalConfig).

  Returns:
    An EvalResult that can be used with the TFMA visualization functions.

  Raises:
    ValueError: If the file_format is unknown to us.
  """
  _assert_tensorflow_version()

  if eval_shared_model is not None:
    eval_shared_models = [eval_shared_model]

  if eval_config is None:
    if output_path is None:
      output_path = tempfile.mkdtemp()
    if not tf.io.gfile.exists(output_path):
      tf.io.gfile.makedirs(output_path)
    disabled_outputs = None
    if not write_config:
      disabled_outputs = [_EVAL_CONFIG_FILE]
    model_specs = []
    for m in eval_shared_models:
      example_weight_key = m.example_weight_key
      example_weight_keys = {}
      if example_weight_key and isinstance(example_weight_key, dict):
        example_weight_keys = example_weight_key
        example_weight_key = ''
      model_specs.append(
          config.ModelSpec(
              location=m.model_path,
              example_weight_key=example_weight_key,
              example_weight_keys=example_weight_keys))
    slicing_specs = None
    if slice_spec:
      slicing_specs = [s.to_proto() for s in slice_spec]
    options = config.Options()
    options.compute_confidence_intervals.value = compute_confidence_intervals
    options.k_anonymization_count.value = k_anonymization_count
    if desired_batch_size:
      options.desired_batch_size.value = desired_batch_size
    eval_config = config.EvalConfig(
        input_data_specs=[
            config.InputDataSpec(
                location=data_location, file_format=file_format)
        ],
        model_specs=model_specs,
        output_data_specs=[
            config.OutputDataSpec(
                default_location=output_path, disabled_outputs=disabled_outputs)
        ],
        slicing_specs=slicing_specs,
        options=options)

  if len(eval_config.input_data_specs) != 1:
    raise NotImplementedError(
        'multiple input_data_specs are not yet supported.')
  if len(eval_config.model_specs) != 1:
    raise NotImplementedError('multiple model_specs are not yet supported.')
  if len(eval_config.output_data_specs) != 1:
    raise NotImplementedError(
        'multiple output_data_specs are not yet supported.')

  with beam.Pipeline(options=pipeline_options) as p:
    if (not eval_config.input_data_specs[0].file_format or
        eval_config.input_data_specs[0].file_format == 'tfrecords'):
      data = p | 'ReadFromTFRecord' >> beam.io.ReadFromTFRecord(
          file_pattern=eval_config.input_data_specs[0].location,
          compression_type=beam.io.filesystem.CompressionTypes.AUTO)
    elif eval_config.input_data_specs[0].file_format == 'text':
      data = p | 'ReadFromText' >> beam.io.textio.ReadFromText(
          eval_config.input_data_specs[0].location)
    else:
      raise ValueError('unknown file_format: {}'.format(
          eval_config.input_data_specs[0].file_format))

    # pylint: disable=no-value-for-parameter
    _ = (
        data
        | 'ExtractEvaluateAndWriteResults' >> ExtractEvaluateAndWriteResults(
            eval_config=eval_config,
            eval_shared_models=eval_shared_models,
            extractors=extractors,
            evaluators=evaluators,
            writers=writers))
    # pylint: enable=no-value-for-parameter

  # TODO(b/141016373): Add support for multiple models.
  return load_eval_result(eval_config.output_data_specs[0].default_location)


def single_model_analysis(
    model_location: Text,
    data_location: Text,
    output_path: Text = None,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None) -> EvalResult:
  """Run model analysis for a single model on a single data set.

  This is a convenience wrapper around run_model_analysis for a single model
  with a single data set. For more complex use cases, use
  tfma.run_model_analysis.

  Args:
    model_location: Path to the export eval saved model.
    data_location: The location of the data files.
    output_path: The directory to output metrics and results to. If None, we use
      a temporary directory.
    slice_spec: A list of tfma.slicer.SingleSliceSpec.

  Returns:
    An EvalResult that can be used with the TFMA visualization functions.
  """
  # Get working_dir ready.
  if output_path is None:
    output_path = tempfile.mkdtemp()
  if not tf.io.gfile.exists(output_path):
    tf.io.gfile.makedirs(output_path)

  eval_config = config.EvalConfig(
      input_data_specs=[config.InputDataSpec(location=data_location)],
      model_specs=[config.ModelSpec(location=model_location)],
      output_data_specs=[config.OutputDataSpec(default_location=output_path)],
      slicing_specs=[s.to_proto() for s in slice_spec])

  return run_model_analysis(
      eval_config=eval_config,
      eval_shared_models=[
          default_eval_shared_model(eval_saved_model_path=model_location)
      ])


def multiple_model_analysis(model_locations: List[Text], data_location: Text,
                            **kwargs) -> EvalResults:
  """Run model analysis for multiple models on the same data set.

  Args:
    model_locations: A list of paths to the export eval saved model.
    data_location: The location of the data files.
    **kwargs: The args used for evaluation. See tfma.single_model_analysis() for
      details.

  Returns:
    A tfma.EvalResults containing all the evaluation results with the same order
    as model_locations.
  """
  results = []
  for m in model_locations:
    results.append(single_model_analysis(m, data_location, **kwargs))
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
    results.append(single_model_analysis(model_location, d, **kwargs))
  return EvalResults(results, constants.DATA_CENTRIC_MODE)
