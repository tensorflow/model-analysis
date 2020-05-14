# Lint as: python3
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

# TODO(b/149126671): Put ValidationResultsWriter in a separate file.

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import os
import tempfile

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Text, Tuple, Union

from absl import logging
import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_constants
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator_v2
from tensorflow_model_analysis.extractors import batched_input_extractor
from tensorflow_model_analysis.extractors import batched_predict_extractor_v2
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import input_extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import predict_extractor_v2
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.extractors import tflite_predict_extractor
from tensorflow_model_analysis.extractors import unbatch_extractor
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.validators import validator
from tensorflow_model_analysis.writers import eval_config_writer
from tensorflow_model_analysis.writers import metrics_and_plots_serialization
from tensorflow_model_analysis.writers import metrics_plots_and_validations_writer
from tensorflow_model_analysis.writers import writer
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record
from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(pachristopher): After TFMA is released, enable batched extractors by
# default.
_ENABLE_BATCHED_EXTRACTORS = False


def _assert_tensorflow_version():
  """Check that we're using a compatible TF version."""
  # Fail with a clear error in case we are not using a compatible TF version.
  major, minor, _ = tf.version.VERSION.split('.')
  if (int(major) not in (1, 2)) or (int(major) == 1 and int(minor) < 15):
    raise RuntimeError(
        'Tensorflow version >= 1.15, < 3 is required. Found (%s). Please '
        'install the latest 1.x or 2.x version from '
        'https://github.com/tensorflow/tensorflow. ' % tf.version.VERSION)
  if int(major) == 2:
    logging.warning(
        'Tensorflow version (%s) found. Note that TFMA support for TF 2.0 '
        'is currently in beta', tf.version.VERSION)


def _is_legacy_eval(
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels],
    eval_config: Optional[config.EvalConfig]):
  """Returns True if legacy evaluation is being used."""
  # A legacy evaluation is an evalution that uses only a single EvalSharedModel,
  # has no tags (or uses "eval" as its tag), and does not specify an eval_config
  # (or specifies an eval_config with no metrics). The legacy evaluation is
  # based on using add_metrics_callbacks to create a modified version of the
  # graph saved with an EvalSavedModel. The newer version of evaluation supports
  # both add_metrics_callbacks as well as metrics defined in MetricsSpecs inside
  # of EvalConfig. The newer version works with both "eval" and serving models
  # and also supports multi-model evaluation. This function is used by code to
  # support backwards compatibility for callers that have not updated to use the
  # new EvalConfig.
  return (eval_shared_model and not isinstance(eval_shared_model, dict) and
          not isinstance(eval_shared_model, list) and
          ((not eval_shared_model.model_loader.tags or
            eval_constants.EVAL_TAG in eval_shared_model.model_loader.tags) and
           (not eval_config or not eval_config.metrics_specs)))


def _default_eval_config(eval_shared_models: List[types.EvalSharedModel],
                         slice_spec: Optional[List[slicer.SingleSliceSpec]],
                         write_config: Optional[bool],
                         compute_confidence_intervals: Optional[bool],
                         min_slice_size: int):
  """Creates default EvalConfig (for use in legacy evaluations)."""
  model_specs = []
  for shared_model in eval_shared_models:
    example_weight_key = shared_model.example_weight_key
    example_weight_keys = {}
    if example_weight_key and isinstance(example_weight_key, dict):
      example_weight_keys = example_weight_key
      example_weight_key = ''
    model_specs.append(
        config.ModelSpec(
            name=shared_model.model_name,
            example_weight_key=example_weight_key,
            example_weight_keys=example_weight_keys))
  slicing_specs = None
  if slice_spec:
    slicing_specs = [s.to_proto() for s in slice_spec]
  options = config.Options()
  options.compute_confidence_intervals.value = compute_confidence_intervals
  options.min_slice_size.value = min_slice_size
  if not write_config:
    options.disabled_outputs.values.append(eval_config_writer.EVAL_CONFIG_FILE)
  return config.EvalConfig(
      model_specs=model_specs, slicing_specs=slicing_specs, options=options)


def _model_types(
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels]
) -> Optional[Set[Text]]:
  """Returns model types associated with given EvalSharedModels."""
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  if not eval_shared_models:
    return None
  else:
    return set([m.model_type for m in eval_shared_models])


def _update_eval_config_with_defaults(
    eval_config: config.EvalConfig,
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels]
) -> config.EvalConfig:
  """Returns updated eval config with default values."""
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  maybe_add_baseline = eval_shared_models and len(eval_shared_models) == 2

  return config.update_eval_config_with_defaults(
      eval_config, maybe_add_baseline=maybe_add_baseline)


MetricsForSlice = metrics_for_slice_pb2.MetricsForSlice


def load_metrics(output_path: Text) -> List[MetricsForSlice]:
  """Read and deserialize the MetricsForSlice records."""
  records = []
  filepath = os.path.join(output_path, constants.METRICS_KEY)
  if not tf.io.gfile.exists(filepath):
    filepath = output_path  # Allow full file to be passed.
  for record in tf.compat.v1.python_io.tf_record_iterator(filepath):
    records.append(MetricsForSlice.FromString(record))
  return records


PlotsForSlice = metrics_for_slice_pb2.PlotsForSlice


def load_plots(output_path: Text) -> List[PlotsForSlice]:
  """Read and deserialize the PlotsForSlice records."""
  records = []
  filepath = os.path.join(output_path, constants.PLOTS_KEY)
  if not tf.io.gfile.exists(filepath):
    filepath = output_path  # Allow full file to be passed.
  for record in tf.compat.v1.python_io.tf_record_iterator(filepath):
    records.append(PlotsForSlice.FromString(record))
  return records


# Define types here to avoid type errors between OSS and internal code.
ValidationResult = validation_result_pb2.ValidationResult


def load_validation_result(output_path: Text) -> Optional[ValidationResult]:
  """Read and deserialize the ValidationResult."""
  validation_records = []
  filepath = os.path.join(output_path, constants.VALIDATIONS_KEY)
  if not tf.io.gfile.exists(filepath):
    filepath = output_path  # Allow full file to be passed.
  for record in tf.compat.v1.python_io.tf_record_iterator(filepath):
    validation_records.append(ValidationResult.FromString(record))
  if validation_records:
    assert len(validation_records) == 1
    return validation_records[0]


_Plot = Dict[Text, Any]
_Metrics = Dict[Text, Any]
_MetricsBySubKey = Dict[Text, _Metrics]
_MetricsByOutputName = Dict[Text, Dict[Text, Dict[Text, _MetricsBySubKey]]]


class EvalResult(
    NamedTuple('EvalResult',
               [('slicing_metrics', List[Tuple[slicer.SliceKeyType,
                                               _MetricsByOutputName]]),
                ('plots', List[Tuple[slicer.SliceKeyType, _Plot]]),
                ('config', config.EvalConfig), ('data_location', Text),
                ('file_format', Text), ('model_location', Text)])):
  """Class for the result of single model analysis run.

  Attributes:
    slicing_metrics: Nested dictionary representing metrics for different
      configurations as defined by MetricKey in metrics_for_slice.proto. The
      levels corresponds to output name, sub key, metric name and metric value
      in this order. The sub key is an encoding of class_id, top_k, and k
      values. Note that MetricValue uses oneof, so metric values will always
      contain only a single key representing the type in the oneof and the
      actual metric value is in the value.
    plots: List of slice-plot pairs.
    config: The config containing slicing and metrics specification.
    data_location: Optional location for data used with config.
    file_format: Optional format for data used with config.
    model_location: Optional location(s) for model(s) used with config.
  """

  def get_metrics(self,
                  slice_name: slicer.SliceKeyType = (),
                  output_name: Text = '',
                  class_id: Optional[int] = None,
                  k: Optional[int] = None,
                  top_k: Optional[int] = None) -> Union[_Metrics, None]:
    """Get metric names and values for a slice.

    Args:
      slice_name: A tuple of the form (column, value), indicating which slice to
        get metrics from. Optional; if excluded, return overall metrics.
      output_name: The name of the output. Optional, only used for multi-output
        models.
      class_id: Used with multi-class metrics to identify a specific class ID.
      k: Used with multi-class metrics to identify the kth predicted value.
      top_k: Used with multi-class and ranking metrics to identify top-k
        predicted values.

    Returns:
      Dictionary containing metric names and values for the specified slice.
    """

    sub_key = metric_types.SubKey(class_id, k, top_k)

    def equals_slice_name(slice_key):
      if not slice_key:
        return not slice_name
      else:
        return slice_key == slice_name

    for slicing_metric in self.slicing_metrics:
      slice_key = slicing_metric[0]
      slice_val = slicing_metric[1]
      if equals_slice_name(slice_key):
        return slice_val[output_name][str(sub_key)]

    # if slice could not be found, return None
    return None

  def get_metrics_for_all_slices(
      self,
      output_name: Text = '',
      class_id: Optional[int] = None,
      k: Optional[int] = None,
      top_k: Optional[int] = None) -> Dict[Text, _Metrics]:
    """Get metric names and values for every slice.

    Args:
      output_name: The name of the output (optional, only used for multi-output
        models).
      class_id: Used with multi-class metrics to identify a specific class ID.
      k: Used with multi-class metrics to identify the kth predicted value.
      top_k: Used with multi-class and ranking metrics to identify top-k
        predicted values.

    Returns:
      Dictionary mapping slices to metric names and values.
    """

    sub_key = metric_types.SubKey(class_id, k, top_k)

    sliced_metrics = {}
    for slicing_metric in self.slicing_metrics:
      slice_name = slicing_metric[0]
      metrics = slicing_metric[1][output_name][str(sub_key)]
      sliced_metrics[slice_name] = {
          metric_name: metric_value
          for metric_name, metric_value in metrics.items()
      }
    return sliced_metrics  # pytype: disable=bad-return-type

  def get_slices(self) -> Sequence[Text]:
    """Get names of slices.

    Returns:
      List of slice names.
    """

    return [slicing_metric[0] for slicing_metric in self.slicing_metrics]  # pytype: disable=bad-return-type


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
  eval_config, data_location, file_format, model_locations = (
      eval_config_writer.load_eval_run(output_path))
  metrics_proto_list = (
      metrics_and_plots_serialization.load_and_deserialize_metrics(
          path=os.path.join(output_path, constants.METRICS_KEY),
          model_name=model_name))
  plots_proto_list = (
      metrics_and_plots_serialization.load_and_deserialize_plots(
          path=os.path.join(output_path, constants.PLOTS_KEY)))

  if model_name is None:
    model_location = list(model_locations.values())[0]
  else:
    model_location = model_locations[model_name]
  return EvalResult(
      slicing_metrics=metrics_proto_list,
      plots=plots_proto_list,
      config=eval_config,
      data_location=data_location,
      file_format=file_format,
      model_location=model_location)


def default_eval_shared_model(
    eval_saved_model_path: Text,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    include_default_metrics: Optional[bool] = True,
    example_weight_key: Optional[Union[Text, Dict[Text, Text]]] = None,
    additional_fetches: Optional[List[Text]] = None,
    blacklist_feature_fetches: Optional[List[Text]] = None,
    tags: Optional[List[Text]] = None,
    model_name: Text = '',
    eval_config: Optional[config.EvalConfig] = None) -> types.EvalSharedModel:
  """Returns default EvalSharedModel.

  Args:
    eval_saved_model_path: Path to EvalSavedModel.
    add_metrics_callbacks: Optional list of callbacks for adding additional
      metrics to the graph (see EvalSharedModel for more information on how to
      configure additional metrics). Metrics for example count and example
      weights will be added automatically.
    include_default_metrics: True to include the default metrics that are part
      of the saved model graph during evaluation. Note that
      eval_config.options.include_default_metrics must also be true.
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
    model_name: Optional name of the model being created (should match
      ModelSpecs.name). The name should only be provided if multiple models are
      being evaluated.
    eval_config: Eval config. Only used for setting default tags.
  """
  if not eval_config:
    model_type = constants.TF_ESTIMATOR
    if tags is None:
      tags = [eval_constants.EVAL_TAG]
  else:
    model_spec = model_util.get_model_spec(eval_config, model_name)
    if not model_spec:
      raise ValueError('ModelSpec for model name {} not found in EvalConfig: '
                       'config={}'.format(model_name, eval_config))
    model_type = model_util.get_model_type(model_spec, eval_saved_model_path,
                                           tags)
    if tags is None:
      # Default to serving unless estimator is used.
      if model_type == constants.TF_ESTIMATOR:
        tags = [eval_constants.EVAL_TAG]
      else:
        tags = [tf.saved_model.SERVING]

  # Backwards compatibility for legacy add_metrics_callbacks implementation.
  if model_type == constants.TF_ESTIMATOR and eval_constants.EVAL_TAG in tags:
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
      model_name=model_name,
      model_type=model_type,
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
              model_type=model_type,
              tags=tags)))


def default_extractors(  # pylint: disable=invalid-name
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    eval_config: config.EvalConfig = None,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    materialize: Optional[bool] = True,
    enable_batched_extractors: Optional[bool] = False,
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None,
) -> List[extractor.Extractor]:
  """Returns the default extractors for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Shared model (single-model evaluation) or list of shared
      models (multi-model evaluation). Required unless the predictions are
      provided alongside of the features (i.e. model-agnostic evaluations).
    eval_config: Eval config.
    slice_spec: Deprecated (use EvalConfig).
    materialize: True to have extractors create materialized output.
    enable_batched_extractors: True if batched extractors should be used.
    tensor_adapter_config: Tensor adapter config which specifies how to obtain
      tensors from the Arrow RecordBatch. If None, we feed the raw examples to
      the model.

  Raises:
    NotImplementedError: If eval_config contains mixed serving and eval models.
  """
  if eval_config is not None:
    eval_config = _update_eval_config_with_defaults(eval_config,
                                                    eval_shared_model)
    slice_spec = [
        slicer.SingleSliceSpec(spec=spec) for spec in eval_config.slicing_specs
    ]

  if _is_legacy_eval(eval_shared_model, eval_config):
    # Backwards compatibility for previous add_metrics_callbacks implementation.
    return [
        predict_extractor.PredictExtractor(
            eval_shared_model, materialize=materialize),
        slice_key_extractor.SliceKeyExtractor(
            slice_spec, materialize=materialize)
    ]
  elif eval_shared_model:
    model_types = _model_types(eval_shared_model)
    eval_shared_models = model_util.verify_and_update_eval_shared_models(
        eval_shared_model)

    if not model_types.issubset(constants.VALID_MODEL_TYPES):
      raise NotImplementedError(
          'model type must be one of: {}. evalconfig={}'.format(
              str(constants.VALID_MODEL_TYPES), eval_config))
    if model_types == set([constants.TF_LITE]):
      return [
          input_extractor.InputExtractor(eval_config=eval_config),
          tflite_predict_extractor.TFLitePredictExtractor(
              eval_config=eval_config, eval_shared_model=eval_shared_model),
          slice_key_extractor.SliceKeyExtractor(
              slice_spec, materialize=materialize)
      ]
    elif constants.TF_LITE in model_types:
      raise NotImplementedError(
          'support for mixing tf_lite and non-tf_lite models is not '
          'implemented: eval_config={}'.format(eval_config))

    elif (eval_config and model_types == set([constants.TF_ESTIMATOR]) and
          all(eval_constants.EVAL_TAG in m.model_loader.tags
              for m in eval_shared_models)):
      return [
          predict_extractor.PredictExtractor(
              eval_shared_model,
              materialize=materialize,
              eval_config=eval_config),
          slice_key_extractor.SliceKeyExtractor(
              slice_spec, materialize=materialize)
      ]
    elif (eval_config and constants.TF_ESTIMATOR in model_types and
          any(eval_constants.EVAL_TAG in m.model_loader.tags
              for m in eval_shared_models)):
      raise NotImplementedError(
          'support for mixing eval and non-eval estimator models is not '
          'implemented: eval_config={}'.format(eval_config))
    else:
      if enable_batched_extractors:
        return [
            batched_input_extractor.BatchedInputExtractor(
                eval_config=eval_config),
            batched_predict_extractor_v2.BatchedPredictExtractor(
                eval_config=eval_config,
                eval_shared_model=eval_shared_model,
                tensor_adapter_config=tensor_adapter_config),
            unbatch_extractor.UnbatchExtractor(),
            slice_key_extractor.SliceKeyExtractor(
                slice_spec, materialize=materialize)
        ]
      else:
        return [
            input_extractor.InputExtractor(eval_config=eval_config),
            predict_extractor_v2.PredictExtractor(
                eval_config=eval_config, eval_shared_model=eval_shared_model),
            slice_key_extractor.SliceKeyExtractor(
                slice_spec, materialize=materialize)
        ]
  else:
    if enable_batched_extractors:
      return [
          batched_input_extractor.BatchedInputExtractor(
              eval_config=eval_config),
          unbatch_extractor.UnbatchExtractor(),
          slice_key_extractor.SliceKeyExtractor(
              slice_spec, materialize=materialize)
      ]
    else:
      return [
          input_extractor.InputExtractor(eval_config=eval_config),
          slice_key_extractor.SliceKeyExtractor(
              slice_spec, materialize=materialize)
      ]


def default_evaluators(  # pylint: disable=invalid-name
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    eval_config: config.EvalConfig = None,
    schema: Optional[schema_pb2.Schema] = None,
    compute_confidence_intervals: Optional[bool] = False,
    min_slice_size: int = 1,
    serialize: bool = False,
    random_seed_for_testing: Optional[int] = None) -> List[evaluator.Evaluator]:
  """Returns the default evaluators for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Optional shared model (single-model evaluation) or list
      of shared models (multi-model evaluation). Only required if there are
      metrics to be computed in-graph using the model.
    eval_config: Eval config.
    schema: A schema to use for customizing default evaluators.
    compute_confidence_intervals: Deprecated (use eval_config).
    min_slice_size: Deprecated (use eval_config).
    serialize: Deprecated.
    random_seed_for_testing: Provide for deterministic tests only.
  """
  disabled_outputs = []
  if eval_config:
    eval_config = _update_eval_config_with_defaults(eval_config,
                                                    eval_shared_model)
    disabled_outputs = eval_config.options.disabled_outputs.values
    if _model_types(eval_shared_model) == set([constants.TF_LITE]):
      # no in-graph metrics present when tflite is used.
      if eval_shared_model:
        if isinstance(eval_shared_model, dict):
          eval_shared_model = {
              k: v._replace(include_default_metrics=False)
              for k, v in eval_shared_model.items()
          }
        elif isinstance(eval_shared_model, list):
          eval_shared_model = [
              v._replace(include_default_metrics=False)
              for v in eval_shared_model
          ]
        else:
          eval_shared_model = eval_shared_model._replace(
              include_default_metrics=False)
  if (constants.METRICS_KEY in disabled_outputs and
      constants.PLOTS_KEY in disabled_outputs):
    return []
  if _is_legacy_eval(eval_shared_model, eval_config):
    # Backwards compatibility for previous add_metrics_callbacks implementation.
    if eval_config is not None:
      if eval_config.options.HasField('compute_confidence_intervals'):
        compute_confidence_intervals = (
            eval_config.options.compute_confidence_intervals.value)
      if eval_config.options.HasField('min_slice_size'):
        min_slice_size = eval_config.options.min_slice_size.value
    return [
        metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(
            eval_shared_model,
            compute_confidence_intervals=compute_confidence_intervals,
            min_slice_size=min_slice_size,
            serialize=serialize,
            random_seed_for_testing=random_seed_for_testing)
    ]
  else:
    return [
        metrics_and_plots_evaluator_v2.MetricsAndPlotsEvaluator(
            eval_config=eval_config,
            eval_shared_model=eval_shared_model,
            schema=schema,
            random_seed_for_testing=random_seed_for_testing)
    ]


def default_writers(
    output_path: Optional[Text],
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    eval_config: Optional[config.EvalConfig] = None,
    display_only_data_location: Optional[Text] = None,
    display_only_file_format: Optional[Text] = None
) -> List[writer.Writer]:  # pylint: disable=invalid-name
  """Returns the default writers for use in WriteResults.

  Args:
    output_path: Output path.
    eval_shared_model: Optional shared model (single-model evaluation) or list
      of shared models (multi-model evaluation). Only required if legacy
      add_metrics_callbacks are used.
    eval_config: Optional eval config for writing out config along with results.
    display_only_data_location: Optional path indicating where the examples were
      read from. This is used only for display purposes - data will not actually
      be read from this path.
    display_only_file_format: Optional format of the examples. This is used only
      for display purposes.
  """
  writers = []

  add_metric_callbacks = []
  # The add_metric_callbacks are used in the metrics and plots serialization
  # code to post process the metric data by calling populate_stats_and_pop.
  # While both the legacy (V1) and new (V2) evaluation implementations support
  # EvalSavedModels using add_metric_callbacks, this particular code is only
  # required for the legacy evaluation based on the MetricsAndPlotsEvaluator.
  # The V2 MetricsAndPlotsEvaluator output requires no additional processing.
  # Since the V1 code only supports a single EvalSharedModel, we only set the
  # add_metrics_callbacks if a dict is not passed.
  if (eval_shared_model and not isinstance(eval_shared_model, dict) and
      not isinstance(eval_shared_model, list)):
    add_metric_callbacks = eval_shared_model.add_metrics_callbacks

  if eval_config:
    model_locations = {}
    eval_shared_models = model_util.verify_and_update_eval_shared_models(
        eval_shared_model)
    for v in (eval_shared_models or [None]):
      k = '' if v is None else v.model_name
      model_locations[k] = ('<unknown>' if v is None or v.model_path is None
                            else v.model_path)
    writers.append(
        eval_config_writer.EvalConfigWriter(
            output_path,
            eval_config=eval_config,
            data_location=display_only_data_location,
            file_format=display_only_file_format,
            model_locations=model_locations))

  output_paths = {
      constants.METRICS_KEY:
          os.path.join(output_path, constants.METRICS_KEY),
      constants.PLOTS_KEY:
          os.path.join(output_path, constants.PLOTS_KEY),
      constants.VALIDATIONS_KEY:
          os.path.join(output_path, constants.VALIDATIONS_KEY)
  }
  writers.append(
      metrics_plots_and_validations_writer.MetricsPlotsAndValidationsWriter(
          output_paths=output_paths,
          add_metrics_callbacks=add_metric_callbacks))
  return writers


@beam.ptransform_fn
# TODO(b/156538355): Find out why str is also required instead of just bytes
#   after adding types.Extracts.
@beam.typehints.with_input_types(Union[bytes, str, types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
def InputsToExtracts(  # pylint: disable=invalid-name
    inputs: beam.pvalue.PCollection):
  """Converts serialized inputs (e.g. examples) to Extracts if not already."""

  def to_extracts(x: Union[bytes, str, types.Extracts]) -> types.Extracts:
    result = {}
    if isinstance(x, dict):
      result.update(x)
    else:
      result[constants.INPUT_KEY] = x
    return result

  return inputs | 'AddInputKey' >> beam.Map(to_extracts)


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[bytes, pa.RecordBatch])
@beam.typehints.with_output_types(types.Extracts)
def BatchedInputsToExtracts(  # pylint: disable=invalid-name
    batched_inputs: beam.pvalue.PCollection):
  """Converts Arrow RecordBatch inputs to Extracts."""

  def to_extracts(x: Union[bytes, pa.RecordBatch]) -> types.Extracts:
    result = {}
    if isinstance(x, dict):
      result.update(x)
    else:
      result[constants.ARROW_RECORD_BATCH_KEY] = x
    return result

  return batched_inputs | 'AddArrowRecordBatchKey' >> beam.Map(to_extracts)


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


def is_batched_input(eval_shared_model: Optional[
    types.MaybeMultipleEvalSharedModels] = None,
                     eval_config: config.EvalConfig = None) -> bool:
  """Returns true if batched input should be used.

   We will keep supporting the legacy unbatched V1 PredictExtractor as it parses
   the features and labels, and is the only solution currently that allows for
   slicing on transformed features. Eventually we should have support for
   transformed features via keras preprocessing layers.

  Args:
    eval_shared_model: Shared model (single-model evaluation) or list of shared
      models (multi-model evaluation). Required unless the predictions are
      provided alongside of the features (i.e. model-agnostic evaluations).
    eval_config: Eval config.

  Returns:
    A boolean indicating if batched extractors should be used.
  """
  if _is_legacy_eval(eval_shared_model, eval_config):
    return False
  elif eval_shared_model:
    model_types = _model_types(eval_shared_model)
    eval_shared_models = model_util.verify_and_update_eval_shared_models(
        eval_shared_model)
    if model_types == set([constants.TF_LITE]):
      return False
    elif (eval_config and model_types == set([constants.TF_ESTIMATOR]) and
          all(eval_constants.EVAL_TAG in m.model_loader.tags
              for m in eval_shared_models)):
      return False
  return True


@beam.ptransform_fn
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def ExtractEvaluateAndWriteResults(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    eval_config: config.EvalConfig = None,
    extractors: Optional[List[extractor.Extractor]] = None,
    evaluators: Optional[List[evaluator.Evaluator]] = None,
    writers: Optional[List[writer.Writer]] = None,
    output_path: Optional[Text] = None,
    display_only_data_location: Optional[Text] = None,
    display_only_file_format: Optional[Text] = None,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    write_config: Optional[bool] = True,
    compute_confidence_intervals: Optional[bool] = False,
    min_slice_size: int = 1,
    random_seed_for_testing: Optional[int] = None,
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None,
    schema: Optional[schema_pb2.Schema] = None) -> beam.pvalue.PDone:
  """PTransform for performing extraction, evaluation, and writing results.

  Users who want to construct their own Beam pipelines instead of using the
  lightweight run_model_analysis functions should use this PTransform.

  Example usage:
    eval_config = tfma.EvalConfig(slicing_specs=[...], metrics_specs=[...])
    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=model_location, eval_config=eval_config)
    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
           | 'ExtractEvaluateAndWriteResults' >>
           tfma.ExtractEvaluateAndWriteResults(
               eval_shared_model=eval_shared_model,
               eval_config=eval_config,
               ...))
    result = tfma.load_eval_result(output_path=output_path)
    tfma.view.render_slicing_metrics(result)

  Note that the exact serialization format is an internal implementation detail
  and subject to change. Users should only use the TFMA functions to write and
  read the results.

  Args:
    examples: PCollection of input examples or Arrow Record batches. Examples
      can be any format the model accepts (e.g. string containing CSV row,
      TensorFlow.Example, etc). If the examples are in the form of a dict it
      will be assumed that input is already in the form of tfma.Extracts with
      examples stored under tfma.INPUT_KEY (any other keys will be passed along
      unchanged to downstream extractors and evaluators).
    eval_shared_model: Optional shared model (single-model evaluation) or list
      of shared models (multi-model evaluation). Only required if needed by
      default extractors, evaluators, or writers and for display purposes of the
      model path.
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
    output_path: Path to output results to (config file, metrics, plots, etc).
    display_only_data_location: Optional path indicating where the examples were
      read from. This is used only for display purposes - data will not actually
      be read from this path.
    display_only_file_format: Optional format of the examples. This is used only
      for display purposes.
    slice_spec: Deprecated (use EvalConfig).
    write_config: Deprecated (use EvalConfig).
    compute_confidence_intervals: Deprecated (use EvalConfig).
    min_slice_size: Deprecated (use EvalConfig).
    random_seed_for_testing: Provide for deterministic tests only.
    tensor_adapter_config: Tensor adapter config which specifies how to obtain
      tensors from the Arrow RecordBatch. If None, we feed the raw examples to
      the model.
    schema: A schema to use for customizing evaluators.

  Raises:
    ValueError: If EvalConfig invalid or matching Extractor not found for an
      Evaluator.

  Returns:
    PDone.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)

  if eval_config is None:
    eval_config = _default_eval_config(eval_shared_models, slice_spec,
                                       write_config,
                                       compute_confidence_intervals,
                                       min_slice_size)
  else:
    eval_config = _update_eval_config_with_defaults(eval_config,
                                                    eval_shared_model)

  config.verify_eval_config(eval_config)

  if not extractors:
    extractors = default_extractors(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        materialize=False,
        enable_batched_extractors=_ENABLE_BATCHED_EXTRACTORS,
        tensor_adapter_config=tensor_adapter_config)

  if not evaluators:
    evaluators = default_evaluators(
        eval_config=eval_config,
        eval_shared_model=eval_shared_model,
        random_seed_for_testing=random_seed_for_testing,
        schema=schema)

  for v in evaluators:
    evaluator.verify_evaluator(v, extractors)

  if not writers:
    writers = default_writers(
        output_path=output_path,
        eval_shared_model=eval_shared_model,
        eval_config=eval_config,
        display_only_data_location=display_only_data_location,
        display_only_file_format=display_only_file_format)

  # pylint: disable=no-value-for-parameter
  if (_ENABLE_BATCHED_EXTRACTORS and
      is_batched_input(eval_shared_model, eval_config)):
    extracts = examples | 'BatchedInputsToExtracts' >> BatchedInputsToExtracts()
  else:
    extracts = examples | 'InputsToExtracts' >> InputsToExtracts()

  _ = (
      extracts
      | 'ExtractAndEvaluate' >> ExtractAndEvaluate(
          extractors=extractors, evaluators=evaluators)
      | 'WriteResults' >> WriteResults(writers=writers))

  return beam.pvalue.PDone(examples.pipeline)


def run_model_analysis(
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels] = None,
    eval_config: config.EvalConfig = None,
    data_location: Text = '',
    file_format: Text = 'tfrecords',
    output_path: Optional[Text] = None,
    extractors: Optional[List[extractor.Extractor]] = None,
    evaluators: Optional[List[evaluator.Evaluator]] = None,
    writers: Optional[List[writer.Writer]] = None,
    pipeline_options: Optional[Any] = None,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    write_config: Optional[bool] = True,
    compute_confidence_intervals: Optional[bool] = False,
    min_slice_size: int = 1,
    random_seed_for_testing: Optional[int] = None,
    schema: Optional[schema_pb2.Schema] = None,
) -> Union[EvalResult, EvalResults]:
  """Runs TensorFlow model analysis.

  It runs a Beam pipeline to compute the slicing metrics exported in TensorFlow
  Eval SavedModel and returns the results.

  This is a simplified API for users who want to quickly get something running
  locally. Users who wish to create their own Beam pipelines can use the
  Evaluate PTransform instead.

  Args:
    eval_shared_model: Optional shared model (single-model evaluation) or list
      of shared models (multi-model evaluation). Only required if needed by
      default extractors, evaluators, or writers.
    eval_config: Eval config.
    data_location: The location of the data files.
    file_format: The file format of the data, can be either 'text' or
      'tfrecords' for now. By default, 'tfrecords' will be used.
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
    pipeline_options: Optional arguments to run the Pipeline, for instance
      whether to run directly.
    slice_spec: Deprecated (use EvalConfig).
    write_config: Deprecated (use EvalConfig).
    compute_confidence_intervals: Deprecated (use EvalConfig).
    min_slice_size: Deprecated (use EvalConfig).
    random_seed_for_testing: Provide for deterministic tests only.
    schema: Optional tf.Metadata schema of the input data.

  Returns:
    An EvalResult that can be used with the TFMA visualization functions.

  Raises:
    ValueError: If the file_format is unknown to us.
  """
  _assert_tensorflow_version()

  if output_path is None:
    output_path = tempfile.mkdtemp()
  if not tf.io.gfile.exists(output_path):
    tf.io.gfile.makedirs(output_path)

  if eval_config is None:
    eval_shared_models = model_util.verify_and_update_eval_shared_models(
        eval_shared_model)
    eval_config = _default_eval_config(eval_shared_models, slice_spec,
                                       write_config,
                                       compute_confidence_intervals,
                                       min_slice_size)
  else:
    eval_config = _update_eval_config_with_defaults(eval_config,
                                                    eval_shared_model)

  tensor_adapter_config = None
  with beam.Pipeline(options=pipeline_options) as p:
    if file_format == 'tfrecords':
      if (_ENABLE_BATCHED_EXTRACTORS and
          is_batched_input(eval_shared_model, eval_config)):
        tfxio = tf_example_record.TFExampleRecord(
            file_pattern=data_location,
            schema=schema,
            raw_record_column_name=constants.ARROW_INPUT_COLUMN)
        if schema is not None:
          tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
              arrow_schema=tfxio.ArrowSchema(),
              tensor_representations=tfxio.TensorRepresentations())
        data = p | 'ReadFromTFRecordToArrow' >> tfxio.BeamSource()
      else:
        data = p | 'ReadFromTFRecord' >> beam.io.ReadFromTFRecord(
            file_pattern=data_location,
            compression_type=beam.io.filesystem.CompressionTypes.AUTO)
    elif file_format == 'text':
      data = p | 'ReadFromText' >> beam.io.textio.ReadFromText(data_location)
    else:
      raise ValueError('unknown file_format: {}'.format(file_format))

    # pylint: disable=no-value-for-parameter
    _ = (
        data
        | 'ExtractEvaluateAndWriteResults' >> ExtractEvaluateAndWriteResults(
            eval_config=eval_config,
            eval_shared_model=eval_shared_model,
            display_only_data_location=data_location,
            display_only_file_format=file_format,
            output_path=output_path,
            extractors=extractors,
            evaluators=evaluators,
            writers=writers,
            random_seed_for_testing=random_seed_for_testing,
            tensor_adapter_config=tensor_adapter_config,
            schema=schema))
    # pylint: enable=no-value-for-parameter

  if len(eval_config.model_specs) <= 1:
    return load_eval_result(output_path)
  else:
    results = []
    for spec in eval_config.model_specs:
      results.append(load_eval_result(output_path, model_name=spec.name))
    return EvalResults(results, constants.MODEL_CENTRIC_MODE)


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
      slicing_specs=[s.to_proto() for s in slice_spec])

  return run_model_analysis(
      eval_config=eval_config,
      eval_shared_model=default_eval_shared_model(
          eval_saved_model_path=model_location),
      data_location=data_location,
      output_path=output_path)  # pytype: disable=bad-return-type


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
