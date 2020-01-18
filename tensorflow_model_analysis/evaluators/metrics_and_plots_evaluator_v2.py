# Lint as: python3
# Copyright 2019 Google LLC
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
"""Metrics and plots evaluation."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy
import datetime
from typing import Any, Dict, Iterable, List, Optional, Text, Tuple, Type, Union

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.evaluators import eval_saved_model_util
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators import poisson_bootstrap
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.slicer import slicer_lib as slicer

_COMBINER_INPUTS_KEY = '_combiner_inputs'
_DEFAULT_COMBINER_INPUT_KEY = '_default_combiner_input'


def MetricsAndPlotsEvaluator(  # pylint: disable=invalid-name
    eval_config: config.EvalConfig,
    eval_shared_model: Optional[Union[types.EvalSharedModel,
                                      Dict[Text,
                                           types.EvalSharedModel]]] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    run_after: Text = slice_key_extractor.SLICE_KEY_EXTRACTOR_STAGE_NAME
) -> evaluator.Evaluator:
  """Creates an Evaluator for evaluating metrics and plots.

  Args:
    eval_config: Eval config.
    eval_shared_model: Optional shared model (single-model evaluation) or dict
      of shared models keyed by model name (multi-model evaluation). Only
      required if there are metrics to be computed in-graph using the model.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    run_after: Extractor to run after (None means before any extractors).

  Returns:
    Evaluator for evaluating metrics and plots. The output will be stored under
    'metrics' and 'plots' keys.
  """
  eval_shared_models = eval_shared_model
  if eval_shared_models:
    if not isinstance(eval_shared_model, dict):
      eval_shared_models = {'': eval_shared_model}
    # To maintain consistency between settings where single models are used,
    # always use '' as the model name regardless of whether a name is passed.
    if len(eval_shared_models) == 1:
      eval_shared_models = {'': list(eval_shared_models.values())[0]}

  # pylint: disable=no-value-for-parameter
  return evaluator.Evaluator(
      stage_name='EvaluateMetricsAndPlots',
      run_after=run_after,
      ptransform=_EvaluateMetricsAndPlots(
          eval_config=eval_config,
          eval_shared_models=eval_shared_models,
          metrics_key=metrics_key,
          plots_key=plots_key))


# Temporary support for legacy format. This is only required if a
# V1 PredictExtractor is used in place of a V2 PredictExtractor in the pipeline
# (i.e. when running V2 evaluation based on inference done from an eval
# saved_model vs from a serving saved_model).
def _convert_legacy_fpl(
    extracts: types.Extracts,
    example_weight_key: Union[Text, Dict[Text, Text]]) -> types.Extracts:
  """Converts from legacy FPL types to features, labels, predictions."""
  if constants.FEATURES_PREDICTIONS_LABELS_KEY not in extracts:
    return extracts

  remove_node = lambda d: {k: list(v.values())[0] for k, v in d.items()}
  remove_batch = lambda v: v[0] if len(v.shape) > 1 and v.shape[0] == 1 else v
  remove_batches = lambda d: {k: remove_batch(v) for k, v in d.items()}
  remove_default_key = lambda d: list(d.values())[0] if len(d) == 1 else d

  extracts = copy.copy(extracts)
  fpl = extracts.pop(constants.FEATURES_PREDICTIONS_LABELS_KEY)
  features = remove_node(fpl.features)
  example_weights = np.array([1.0])
  if example_weight_key:
    if isinstance(example_weight_key, dict):
      example_weights = {}
      for k, v in example_weight_key.items():
        example_weights[k] = remove_batch(features[v])
    else:
      example_weights = remove_batch(features[example_weight_key])
  labels = remove_default_key(remove_batches(remove_node(fpl.labels)))
  predictions = remove_default_key(remove_batches(remove_node(fpl.predictions)))
  extracts[constants.FEATURES_KEY] = features
  extracts[constants.PREDICTIONS_KEY] = predictions
  extracts[constants.LABELS_KEY] = labels
  extracts[constants.EXAMPLE_WEIGHTS_KEY] = example_weights
  return extracts


def _filter_and_separate_computations(
    computations: metric_types.MetricComputations
) -> Tuple[List[metric_types.MetricComputation],
           List[metric_types.DerivedMetricComputation]]:
  """Filters duplicate computations and separates non-derived and derived.

  All metrics are based on either direct computations using combiners or are
  based on the results of one or more other computations. This code separates
  the two types of computations so that only the combiner based computations are
  passed to the main combiner call and the remainder are processed after those
  combiners have run. Filtering is required because DerivedMetricComputations
  typically include copies of the MetricComputations that they depend on in
  order to avoid having to pre-construct and pass around all the dependencies at
  the time the metrics are constructed. Instead, each derived metric creates a
  version of the metric it depends on and then this code de-dups metrics that
  are identical so only one gets computed.

  Args:
    computations: Computations.

  Returns:
    Tuple of (metric computations, derived metric computations).
  """
  non_derived_computations = []
  derived_computations = []
  types_and_keys = {}
  for c in computations:
    if isinstance(c, metric_types.MetricComputation):
      cls = c.__class__.__name__
      keys = sorted(c.keys)
      if cls in types_and_keys:
        # TODO(mdreves): This assumes the user used unique names for all the
        # keys and classes. This could mask a bug where the same name is
        # accidently used for different metric configurations. Add support for
        # creating a dict config for the computations (similar to keras) and
        # then comparing the configs to ensure the classes are identical.
        if keys == types_and_keys[cls]:
          continue
      types_and_keys[cls] = keys
      non_derived_computations.append(c)
    elif isinstance(c, metric_types.DerivedMetricComputation):
      derived_computations.append(c)
    else:
      raise TypeError('Unsupported metric computation type: {}'.format(c))
  return non_derived_computations, derived_computations


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(List[types.Extracts])
def _GroupByQueryKey(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    query_key: Text,
) -> beam.pvalue.PCollection:
  """PTransform for grouping extracts by a query key.

  Args:
    extracts: Incoming PCollection consisting of extracts.
    query_key: Query key to group extracts by. Must be a member of the dict of
      features stored under tfma.FEATURES_KEY.

  Returns:
    PCollection of lists of extracts where each list is associated with same
    query key.
  """
  missing_query_key_counter = beam.metrics.Metrics.counter(
      constants.METRICS_NAMESPACE, 'missing_query_key')

  def key_by_query_key(
      extracts: types.Extracts,
      query_key: Text) -> Optional[Tuple[Text, types.Extracts]]:
    """Extract the query key from the extract and key by that."""
    value = metric_util.to_scalar(
        util.get_by_keys(
            extracts, [constants.FEATURES_KEY, query_key], optional=True),
        tensor_name=query_key)
    if value is None:
      missing_query_key_counter.inc()
      return ('', extracts)
    return ('{}'.format(value), extracts)

  # pylint: disable=no-value-for-parameter
  return (extracts
          | 'KeyByQueryId' >> beam.Map(key_by_query_key, query_key)
          | 'GroupByKey' >> beam.CombinePerKey(beam.combiners.ToListCombineFn())
          | 'DropQueryId' >> beam.Map(lambda kv: kv[1]))


class _PreprocessorDoFn(beam.DoFn):
  """Do function that computes initial state from extracts.

  The outputs for each preprocessor are stored under the key '_combiner_inputs'
  in the overall extracts returned by this process call. These outputs are
  stored as a list in same order as the computations were passed as input so
  that the combiner can later access them by index. For computations that use
  the default labels, predictions, and example weights as their combiner inputs,
  the list entries will contain None values. A '_default_combiner_inputs'
  extract will also exist (if needed) containing StandardMetricInputs.

  If a FeaturePreprocessor is used the outputs of the preprocessor will be
  combined with the default labels, predictions, and example weights and stored
  in the StandardMetricInputs features value under the _default_combiner_inputs
  key.

  If the incoming data is a list of extracts (i.e. a query_key was used), the
  output will be a single extract with the keys within the extract representing
  the list as processed by the preprocessor. For example, the _slice_key_types
  will be a merger of all unique _slice key_types across the extracts list
  and the _default_combiner_inputs will be a list of StandardMetricInputs (one
  for each example matching the query_key).
  """

  def __init__(self, computations: List[metric_types.MetricComputation]):
    self._computations = computations
    self._evaluate_num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'evaluate_num_instances')
    self._timer = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, '_PreprocessorDoFn_seconds')

  def setup(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.setup()

  def start_bundle(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.start_bundle()

  def finish_bundle(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.finish_bundle()

  def teardown(self):
    for computation in self._computations:
      if computation.preprocessor is not None:
        computation.preprocessor.teardown()

  def process(
      self, extracts: Union[types.Extracts,
                            List[types.Extracts]]) -> Iterable[Any]:
    start_time = datetime.datetime.now()
    self._evaluate_num_instances.inc(1)

    # Assume multiple extracts (i.e. query key used) and reset after if only one
    list_of_extracts = extracts
    if not isinstance(extracts, list):
      list_of_extracts = [extracts]

    use_default_combiner_input = None
    features = None
    combiner_inputs = []
    for computation in self._computations:
      if computation.preprocessor is None:
        combiner_inputs.append(None)
        use_default_combiner_input = True
      elif isinstance(computation.preprocessor,
                      metric_types.FeaturePreprocessor):
        if features is None:
          features = [{} for i in range(len(list_of_extracts))]
        for i, e in enumerate(list_of_extracts):
          for v in computation.preprocessor.process(e):
            features[i].update(v)
        combiner_inputs.append(None)
        use_default_combiner_input = True
      else:
        combiner_inputs.append(next(computation.preprocessor.process(extracts)))

    output = {}
    # Merge the keys for all extracts together.
    slice_key_types = {}
    for e in list_of_extracts:
      for s in e[constants.SLICE_KEY_TYPES_KEY]:
        slice_key_types[s] = True
    output[constants.SLICE_KEY_TYPES_KEY] = list(slice_key_types.keys())
    output[_COMBINER_INPUTS_KEY] = combiner_inputs
    if use_default_combiner_input:
      default_combiner_input = []
      for i, e in enumerate(list_of_extracts):
        if features is not None:
          e = copy.copy(e)
          e.update({constants.FEATURES_KEY: features[i]})
        default_combiner_input.append(
            metric_util.to_standard_metric_inputs(
                e, include_features=features is not None))
      if not isinstance(extracts, list):
        # Not a list, reset to single StandardMetricInput value
        default_combiner_input = default_combiner_input[0]
      output[_DEFAULT_COMBINER_INPUT_KEY] = default_combiner_input
    yield output

    self._timer.update(
        int((datetime.datetime.now() - start_time).total_seconds()))


class _ComputationsCombineFn(beam.combiners.SingleInputTupleCombineFn):
  """Combine function that computes metric using initial state from extracts."""

  def __init__(self,
               computations: List[metric_types.MetricComputation],
               compute_with_sampling: Optional[bool] = False,
               random_seed_for_testing: Optional[int] = None):
    """Init.

    If compute_with_sampling is true a bootstrap resample of the data will be
    performed where each input will be represented in the resample one or more
    times as drawn from Poisson(1). This technically works with small or empty
    batches, but as the technique is an approximation the approximation gets
    better as the number of examples gets larger. If the results themselves are
    empty TFMA will reject the sample. For any samples of a reasonable size, the
    chances of this are exponentially tiny. See "The mathematical fine print"
    section of the blog post linked below.

    See:
    http://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html

    Args:
      computations: List of MetricComputations.
      compute_with_sampling: True to compute with sampling.
      random_seed_for_testing: Seed to use for unit testing.
    """
    super(_ComputationsCombineFn,
          self).__init__(*[c.combiner for c in computations])
    self._compute_with_sampling = compute_with_sampling
    self._random_state = np.random.RandomState(random_seed_for_testing)
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')
    # This keeps track of the number of times the poisson bootstrap encounters
    # an empty set of elements for a slice sample. Should be extremely rare in
    # practice, keeping this counter will help us understand if something is
    # misbehaving.
    self._num_bootstrap_empties = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_bootstrap_empties')

  def add_input(self, accumulator, element):
    elements = [element]
    if self._compute_with_sampling:
      elements = [element] * int(self._random_state.poisson(1, 1))
    if not elements:
      return accumulator

    def get_combiner_input(element, i):
      item = element[_COMBINER_INPUTS_KEY][i]
      if item is None:
        item = element[_DEFAULT_COMBINER_INPUT_KEY]
      return item

    results = []
    for i, (c, a) in enumerate(zip(self._combiners, accumulator)):
      result = c.add_input(a, get_combiner_input(elements[0], i))
      for e in elements[1:]:
        result = c.add_input(result, get_combiner_input(e, i))
      results.append(result)
    return results

  def compact(self, accumulator: Any) -> Any:
    self._num_compacts.inc(1)
    return super(_ComputationsCombineFn, self).compact(accumulator)

  def extract_output(self, accumulator: Any) -> Tuple[Dict[Any, Any]]:
    result = []
    for c, a in zip(self._combiners, accumulator):
      output = c.extract_output(a)
      if not output:
        # Increase a counter for empty bootstrap samples. When sampling is not
        # enabled, this should never be exected. This should only occur when the
        # slice sizes are incredibly small, and seeing large values of this
        # counter is a sign that something has gone wrong.
        self._num_bootstrap_empties.inc(1)
      result.append(output)
    return tuple(result)


@beam.ptransform_fn
@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(Tuple[slicer.SliceKeyType,
                                        Dict[metric_types.MetricKey, Any]])
def _ComputePerSlice(  # pylint: disable=invalid-name
    sliced_extracts: beam.pvalue.PCollection,
    computations: List[metric_types.MetricComputation],
    derived_computations: List[metric_types.DerivedMetricComputation],
    compute_with_sampling: Optional[bool] = False,
    random_seed_for_testing: Optional[int] = None) -> beam.pvalue.PCollection:
  """PTransform for computing, aggregating and combining metrics and plots.

  Args:
    sliced_extracts: Incoming PCollection consisting of slice key and extracts.
    computations: List of MetricComputations.
    derived_computations: List of DerivedMetricComputations.
    compute_with_sampling: True to compute with sampling.
    random_seed_for_testing: Seed to use for unit testing.

  Returns:
    PCollection of (slice key, dict of metrics).
  """
  # TODO(b/123516222): Remove this workaround per discussions in CL/227944001
  sliced_extracts.element_type = beam.typehints.Any

  def convert_and_add_derived_values(
      sliced_results: Tuple[Text, Tuple[Any, ...]],
      derived_computations: List[metric_types.DerivedMetricComputation],
  ) -> Tuple[slicer.SliceKeyType, Dict[metric_types.MetricKey, Any]]:
    """Converts per slice tuple of dicts into single dict and adds derived."""
    result = {}
    for v in sliced_results[1]:
      result.update(v)
    for c in derived_computations:
      result.update(c.result(result))
    # Remove private metrics
    keys = list(result.keys())
    for k in keys:
      if k.name.startswith('_'):
        result.pop(k)
    return (sliced_results[0], result)

  # A fanout of 8 is used here to reduce stragglers that occur during the
  # merger of large datasets such as historgram buckets. This has little effect
  # on the msec profiles, but can impact the wall time and memory usage. If
  # experiencing significantly extended run times due to stragglers, try bumping
  # this to a larger number.
  return (sliced_extracts
          | 'CombinePerSliceKey' >> beam.CombinePerKey(
              _ComputationsCombineFn(
                  computations=computations,
                  compute_with_sampling=compute_with_sampling,
                  random_seed_for_testing=random_seed_for_testing))
          .with_hot_key_fanout(8)
          | 'ConvertAndAddDerivedValues' >> beam.Map(
              convert_and_add_derived_values, derived_computations))


def _filter_by_key_type(
    sliced_metrics_and_plots: Tuple[slicer.SliceKeyType,
                                    Dict[metric_types.MetricKey, Any]],
    key_type: Type[Union[metric_types.MetricKey, metric_types.PlotKey]]
) -> Tuple[slicer.SliceKeyType, Dict[Text, Any]]:
  """Filters metrics and plots by key type."""
  slice_value, metrics_and_plots = sliced_metrics_and_plots
  output = {}
  for k, v in metrics_and_plots.items():
    # PlotKey is a subclass of MetricKey so must check key_type based on PlotKey
    if key_type == metric_types.PlotKey:
      if isinstance(k, metric_types.PlotKey):
        output[k] = v
    else:
      if not isinstance(k, metric_types.PlotKey):
        output[k] = v
  return (slice_value, output)


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[types.Extracts, List[types.Extracts]])
@beam.typehints.with_output_types(evaluator.Evaluation)
def _ComputeMetricsAndPlots(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig,
    metrics_specs: List[config.MetricsSpec],
    eval_shared_models: Optional[Dict[Text, types.EvalSharedModel]] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY) -> evaluator.Evaluation:
  """Computes metrics and plots.

  Args:
    extracts: PCollection of Extracts. If a query_key was used then the
      PCollection will contain a list of extracts.
    eval_config: Eval config.
    metrics_specs: Subset of the metric specs to compute metrics for. If a
      query_key was used all of the metric specs will be for the same query_key.
    eval_shared_models: Optional dict of shared models keyed by model name. Only
      required if there are metrics to be computed in-graph using the model.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.

  Returns:
    Evaluation containing dict of PCollections of (slice_key, results_dict)
    tuples where the dict is keyed by either the metrics_key (e.g. 'metrics') or
    plots_key (e.g. 'plots') depending on what the results_dict contains.
  """
  model_loaders = None
  if eval_shared_models:
    model_loaders = {}
    for k, v in eval_shared_models.items():
      if v.include_default_metrics:
        model_loaders[k] = v.model_loader
  computations, derived_computations = _filter_and_separate_computations(
      metric_specs.to_computations(
          metrics_specs, eval_config=eval_config, model_loaders=model_loaders))
  # Add default metric computations
  if (model_loaders and eval_config and
      (not eval_config.options.HasField('include_default_metrics') or
       eval_config.options.include_default_metrics.value)):
    for model_name, model_loader in model_loaders.items():
      model_types = model_loader.construct_fn(lambda x: None)()
      if model_types.keras_model is not None:
        # TODO(mdreves): Move handling of keras metrics to here.
        pass
      elif model_types.eval_saved_model is not None:
        # Note that there is the possibility for metric naming collisions here
        # (e.g. 'auc' calculated within the EvalSavedModel as well as by AUC
        # metric computation performed outside the model). Currently all the
        # overlapping metrics such as AUC that are computed outside the model
        # are all derived metrics so they will override the metrics calculated
        # by the model which is the desired behavior.
        computations.extend(
            eval_saved_model_util.metric_computations_using_eval_saved_model(
                model_name, model_loader))

  # pylint: disable=no-value-for-parameter

  # Input: Single extract per example (or list of extracts if query_key used)
  #        where each item contains slice keys and other extracts from upstream
  #        extractors (e.g. labels, predictions, etc).
  # Output: Single extract (per example) containing slice keys and initial
  #         combiner state returned from preprocessor. Note that even if a
  #         query_key was used the output is still only a single extract
  #         (though, that extract may contain lists of values (predictions,
  #         labels, etc) in its keys).
  #
  # Note that the output of this step is extracts instead of just a tuple of
  # computation outputs because FanoutSlices takes extracts as input (and in
  # many cases a subset of the extracts themselves are what is fanned out).
  extracts = (
      extracts
      | 'Preprocesss' >> beam.ParDo(_PreprocessorDoFn(computations)))

  # Input: Single extract containing slice keys and initial combiner inputs. If
  #        query_key is used the extract represents multiple examples with the
  #        same query_key, otherwise the extract represents a single example.
  # Output: Tuple (slice key, combiner inputs extracts). Notice that the per
  #         example (or list or examples if query_key used) input extract turns
  #         into n logical extracts, references to which are replicated once per
  #         applicable slice key.
  slices = extracts | 'FanoutSlices' >> slicer.FanoutSlices()

  slices_count = (
      slices
      | 'ExtractSliceKeys' >> beam.Keys()
      | 'CountPerSliceKey' >> beam.combiners.Count.PerElement())

  # Input: Tuple of (slice key, combiner input extracts).
  # Output: Tuple of (slice key, dict of computed metrics/plots). The dicts will
  #         be keyed by MetricKey/PlotKey and the values will be the result
  #         of the associated computations. A given MetricComputation can
  #         perform computations for multiple keys, but the keys should be
  #         unique across computations.
  sliced_metrics_and_plots = (
      slices
      | 'ComputePerSlice' >> poisson_bootstrap.ComputeWithConfidenceIntervals(
          _ComputePerSlice,
          computations=computations,
          derived_computations=derived_computations,
          num_bootstrap_samples=(
              poisson_bootstrap.DEFAULT_NUM_BOOTSTRAP_SAMPLES if
              eval_config.options.compute_confidence_intervals.value else 1)))

  if eval_config.options.k_anonymization_count.value > 1:
    sliced_metrics_and_plots = (
        sliced_metrics_and_plots
        | 'FilteForSmallSlices' >> slicer.FilterOutSlices(
            slices_count, eval_config.options.k_anonymization_count.value))

  sliced_metrics = (
      sliced_metrics_and_plots
      | 'FilterByMetrics' >> beam.Map(_filter_by_key_type,
                                      metric_types.MetricKey))
  sliced_plots = (
      sliced_metrics_and_plots
      | 'FilterByPlots' >> beam.Map(_filter_by_key_type, metric_types.PlotKey))

  # pylint: enable=no-value-for-parameter

  return {metrics_key: sliced_metrics, plots_key: sliced_plots}


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(evaluator.Evaluation)
def _EvaluateMetricsAndPlots(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_config: config.EvalConfig,
    eval_shared_models: Optional[Dict[Text, types.EvalSharedModel]] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY) -> evaluator.Evaluation:
  """Evaluates metrics and plots.

  Args:
    extracts: PCollection of Extracts. The extracts must contain a list of
      slices of type SliceKeyType keyed by tfma.SLICE_KEY_TYPES_KEY as well as
      any extracts required by the metric implementations (typically this will
      include labels keyed by tfma.LABELS_KEY, predictions keyed by
      tfma.PREDICTIONS_KEY, and example weights keyed by
      tfma.EXAMPLE_WEIGHTS_KEY). Usually these will be added by calling the
      default_extractors function.
    eval_config: Eval config.
   eval_shared_models: Optional dict of shared models keyed by model name. Only
     required if there are metrics to be computed in-graph using the model.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.

  Returns:
    Evaluation containing dict of PCollections of (slice_key, results_dict)
    tuples where the dict is keyed by either the metrics_key (e.g. 'metrics') or
    plots_key (e.g. 'plots') depending on what the results_dict contains.
  """
  # Separate metrics based on query_key (which may be None).
  metrics_specs_by_query_key = {}
  for spec in eval_config.metrics_specs:
    if spec.query_key not in metrics_specs_by_query_key:
      metrics_specs_by_query_key[spec.query_key] = []
    metrics_specs_by_query_key[spec.query_key].append(spec)

  # If there are no metrics specs then add an empty one (this is required for
  # cases where only the default metrics from the model are used).
  if not metrics_specs_by_query_key:
    metrics_specs_by_query_key[''] = [config.MetricsSpec()]

  # pylint: disable=no-value-for-parameter

  extracts = extracts | 'ConvertLegacyFPL' >> beam.Map(
      _convert_legacy_fpl, eval_config.model_specs[0].example_weight_key)

  evaluations = {}
  for query_key, metrics_specs in metrics_specs_by_query_key.items():
    query_key_text = query_key if query_key else ''
    if query_key:
      extracts_for_evaluation = (
          extracts
          | 'GroupByQueryKey({})'.format(query_key_text) >>
          _GroupByQueryKey(query_key))
    else:
      extracts_for_evaluation = extracts
    evaluation = (
        extracts_for_evaluation
        | 'ComputeMetricsAndPlots({})'.format(query_key_text) >>
        _ComputeMetricsAndPlots(
            eval_config=eval_config,
            metrics_specs=metrics_specs,
            eval_shared_models=eval_shared_models,
            metrics_key=metrics_key,
            plots_key=plots_key))
    for k, v in evaluation.items():
      if k not in evaluations:
        evaluations[k] = []
      evaluations[k].append(v)

  return evaluator.combine_dict_based_evaluations(evaluations)
