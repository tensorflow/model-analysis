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
"""Metric types."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy
import functools
import inspect

from typing import Any, Callable, Dict, Iterable, List, MutableMapping, NamedTuple, Optional, Text, Type, Union

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf import text_format

# LINT.IfChange


# A separate version from proto is used here because protos are not hashable and
# SerializeToString is not guaranteed to be stable between different binaries.
@functools.total_ordering
class SubKey(
    NamedTuple('SubKey', [('class_id', int), ('k', int), ('top_k', int)])):
  """A SubKey identifies a sub-types of metrics and plots.

  Only one of class_id, k, or top_k can be set at a time.

  Attributes:
    class_id: Used with multi-class metrics to identify a specific class ID.
    k: Used with multi-class metrics to identify the kth predicted value.
    top_k: Used with multi-class and ranking metrics to identify top-k predicted
      values.
  """

  # IfChange (should be preceded by LINT, but cannot nest LINT)
  def __new__(cls,
              class_id: Optional[int] = None,
              k: Optional[int] = None,
              top_k: Optional[int] = None):
    if sum([0 if v is None else 1 for v in (class_id, k, top_k)]) > 1:
      raise ValueError('only one of class_id, k, or top_k should be used: '
                       'class_id={}, k={}, top_k={}'.format(class_id, k, top_k))
    if k is not None and k < 1:
      raise ValueError('attempt to create metric with k < 1: k={}'.format(k))
    if top_k is not None and top_k < 1:
      raise ValueError(
          'attempt to create metric with top_k < 1: top_k={}'.format(top_k))
    return super(SubKey, cls).__new__(cls, class_id, k, top_k)

  # ThenChange(../api/model_eval_lib.py)

  def __eq__(self, other):
    return tuple(self) == other

  def __lt__(self, other):
    # Python3 does not allow comparison of NoneType, remove if present.
    return (tuple(x if x is not None else -1 for x in self) < tuple(
        x if x is not None else -1 for x in other))

  def __hash__(self):
    return hash(tuple(self))

  def __str__(self) -> Text:
    if self.class_id is not None:
      return 'classId:' + str(self.class_id)
    elif self.k is not None:
      return 'k:' + str(self.k)
    elif self.top_k is not None:
      return 'topK:' + str(self.top_k)
    else:
      raise NotImplementedError(
          ('A non-existent SubKey should be represented as None, not as ',
           'SubKey(None, None, None).'))

  def to_proto(self) -> metrics_for_slice_pb2.SubKey:
    """Converts key to proto."""
    sub_key = metrics_for_slice_pb2.SubKey()
    if self.class_id is not None:
      sub_key.class_id.value = self.class_id
    if self.k is not None:
      sub_key.k.value = self.k
    if self.top_k is not None:
      sub_key.top_k.value = self.top_k
    return sub_key

  @staticmethod
  def from_proto(pb: metrics_for_slice_pb2.SubKey) -> Optional['SubKey']:
    """Creates class from proto."""
    class_id = None
    if pb.HasField('class_id'):
      class_id = pb.class_id.value
    k = None
    if pb.HasField('k'):
      k = pb.k.value
    top_k = None
    if pb.HasField('top_k'):
      top_k = pb.top_k.value
    if class_id is None and k is None and top_k is None:
      return None
    else:
      return SubKey(class_id=class_id, k=k, top_k=top_k)


# A separate version from proto is used here because protos are not hashable and
# SerializeToString is not guaranteed to be stable between different binaries.
@functools.total_ordering
class AggregationType(
    NamedTuple('AggregationType', [('micro_average', bool),
                                   ('macro_average', bool),
                                   ('weighted_macro_average', bool)])):
  """AggregationType identifies aggregation types used with AggregationOptions.

  Only one of micro_average, macro_average, or weighted_macro_average can be set
  at a time.

  Attributes:
    micro_average: True of macro averaging used.
    macro_average: True of macro averaging used.
    weighted_macro_average: True of weighted macro averaging used.
  """

  # IfChange (should be preceded by LINT, but cannot nest LINT)
  def __new__(cls,
              micro_average: Optional[bool] = None,
              macro_average: Optional[bool] = None,
              weighted_macro_average: Optional[bool] = None):
    if sum([
        micro_average or False, macro_average or False,
        weighted_macro_average or False
    ]) > 1:
      raise ValueError(
          'only one of micro_average, macro_average, or '
          'weighted_macro_average should be set: micro_average={}, '
          'macro_average={}, weighted_macro_average={}'.format(
              micro_average, macro_average, weighted_macro_average))
    return super(AggregationType,
                 cls).__new__(cls, micro_average, macro_average,
                              weighted_macro_average)

  # ThenChange(../api/model_eval_lib.py)

  def __eq__(self, other):
    return tuple(self) == other

  def __lt__(self, other):
    # Python3 does not allow comparison of NoneType, replace with -1.
    return (tuple(x if x is not None else -1 for x in self) < tuple(
        x if x is not None else -1 for x in other))

  def __hash__(self):
    return hash(tuple(self))

  def __str__(self) -> Text:
    if self.micro_average is not None:
      return 'micro'
    elif self.macro_average is not None:
      return 'macro'
    elif self.weighted_macro_average is not None:
      return 'weighted_macro'
    else:
      raise NotImplementedError(
          ('A non-existent AggregationType should be represented as None, not '
           'as AggregationType(None, None, None).'))

  def to_proto(self) -> metrics_for_slice_pb2.AggregationType:
    """Converts key to proto."""
    aggregration_type = metrics_for_slice_pb2.AggregationType()
    if self.micro_average is not None:
      aggregration_type.micro_average = True
    if self.macro_average is not None:
      aggregration_type.macro_average = True
    if self.weighted_macro_average is not None:
      aggregration_type.weighted_macro_average = True
    return aggregration_type

  @staticmethod
  def from_proto(
      pb: metrics_for_slice_pb2.AggregationType) -> Optional['AggregationType']:
    """Creates class from proto."""
    if pb.micro_average or pb.macro_average or pb.weighted_macro_average:
      return AggregationType(
          micro_average=pb.micro_average or None,
          macro_average=pb.macro_average or None,
          weighted_macro_average=pb.weighted_macro_average or None)
    else:
      return None


# A separate version from proto is used here because protos are not hashable and
# SerializeToString is not guaranteed to be stable between different binaries.
@functools.total_ordering
class MetricKey(
    NamedTuple('MetricKey', [('name', Text), ('model_name', Text),
                             ('output_name', Text), ('sub_key', SubKey),
                             ('aggregation_type', AggregationType),
                             ('is_diff', bool)])):
  """A MetricKey uniquely identifies a metric.

  Attributes:
    name: Metric name. Names starting with '_' are private and will be filtered
      from the final results. Names starting with two underscores, '__' are
      reserved for internal use.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    aggregation_type: Aggregation type.
    is_diff: Optional flag to indicate whether this metrics is a diff metric.
  """

  def __new__(cls,
              name: Text,
              model_name: Text = '',
              output_name: Text = '',
              sub_key: Optional[SubKey] = None,
              aggregation_type: Optional[AggregationType] = None,
              is_diff: Optional[bool] = False):
    return super(MetricKey, cls).__new__(cls, name, model_name, output_name,
                                         sub_key, aggregation_type, is_diff)

  def __eq__(self, other):
    return tuple(self) == other

  def __lt__(self, other):
    # Python3 does not allow comparison of NoneType, remove if present.
    sub_key = self.sub_key if self.sub_key else ()
    other_sub_key = other.sub_key if other.sub_key else ()
    agg_type = self.aggregation_type if self.aggregation_type else ()
    other_agg_type = other.aggregation_type if other.aggregation_type else ()
    is_diff = self.is_diff
    other_is_diff = other.is_diff
    return ((tuple(self[:-3])) < tuple(other[:-3]) and
            sub_key < other_sub_key and agg_type < other_agg_type and
            is_diff < other_is_diff)

  def __hash__(self):
    return hash(tuple(self))

  def __str__(self):
    return text_format.MessageToString(
        self.to_proto(), as_one_line=True, force_colon=True)

  def to_proto(self) -> metrics_for_slice_pb2.MetricKey:
    """Converts key to proto."""
    metric_key = metrics_for_slice_pb2.MetricKey()
    if self.name:
      metric_key.name = self.name
    if self.model_name:
      metric_key.model_name = self.model_name
    if self.output_name:
      metric_key.output_name = self.output_name
    if self.sub_key:
      metric_key.sub_key.CopyFrom(self.sub_key.to_proto())
    if self.aggregation_type:
      metric_key.aggregation_type.CopyFrom(self.aggregation_type.to_proto())
    if self.is_diff:
      metric_key.is_diff = self.is_diff
    return metric_key

  @staticmethod
  def from_proto(pb: metrics_for_slice_pb2.MetricKey) -> 'MetricKey':
    """Configures class from proto."""
    return MetricKey(
        name=pb.name,
        model_name=pb.model_name,
        output_name=pb.output_name,
        sub_key=SubKey.from_proto(pb.sub_key),
        aggregation_type=AggregationType.from_proto(pb.aggregation_type),
        is_diff=pb.is_diff)

  # Generate a copy of the key except that the is_diff is True.
  def make_diff_key(self) -> 'MetricKey':
    return self._replace(is_diff=True)

  # Generate a copy of the key with a different model name and is_diff False.
  def make_baseline_key(self, model_name: Text) -> 'MetricKey':
    return self._replace(model_name=model_name, is_diff=False)


# The output type of a MetricComputation combiner.
MetricsDict = Dict[MetricKey, Any]


# A separate version from proto is used here because protos are not hashable and
# SerializeToString is not guaranteed to be stable between different binaries.
# In addition internally PlotKey is a subclass of MetricKey as each plot is
# stored separately.
class PlotKey(MetricKey):
  """A PlotKey is a metric key that uniquely identifies a plot."""

  def to_proto(self) -> metrics_for_slice_pb2.PlotKey:
    """Converts key to proto."""
    plot_key = metrics_for_slice_pb2.PlotKey()
    if self.name:
      raise ValueError('plot values must be combined into a single proto and'
                       'stored under a plot key without a name')
    if self.model_name:
      plot_key.model_name = self.model_name
    if self.output_name:
      plot_key.output_name = self.output_name
    if self.sub_key:
      plot_key.sub_key.CopyFrom(self.sub_key.to_proto())
    return plot_key

  @staticmethod
  def from_proto(pb: metrics_for_slice_pb2.PlotKey) -> 'PlotKey':
    """Configures class from proto."""
    return PlotKey(
        name='',
        model_name=pb.model_name,
        output_name=pb.output_name,
        sub_key=SubKey.from_proto(pb.sub_key))


# A separate version from proto is used here because protos are not hashable and
# SerializeToString is not guaranteed to be stable between different binaries.
# In addition internally AttributionsKey is a subclass of MetricKey as each
# attribution is stored separately.
class AttributionsKey(MetricKey):
  """An AttributionsKey is a metric key uniquely identifying attributions."""

  def to_proto(self) -> metrics_for_slice_pb2.AttributionsKey:
    """Converts key to proto."""
    attribution_key = metrics_for_slice_pb2.AttributionsKey()
    if self.name:
      attribution_key.name = self.name
    if self.model_name:
      attribution_key.model_name = self.model_name
    if self.output_name:
      attribution_key.output_name = self.output_name
    if self.sub_key:
      attribution_key.sub_key.CopyFrom(self.sub_key.to_proto())
    return attribution_key

  @staticmethod
  def from_proto(
      pb: metrics_for_slice_pb2.AttributionsKey) -> 'AttributionsKey':
    """Configures class from proto."""
    return AttributionsKey(
        name=pb.name,
        model_name=pb.model_name,
        output_name=pb.output_name,
        sub_key=SubKey.from_proto(pb.sub_key))


# LINT.ThenChange(../proto/metrics_for_slice.proto)


class MetricComputation(
    NamedTuple('MetricComputation', [('keys', List[MetricKey]),
                                     ('preprocessor', beam.DoFn),
                                     ('combiner', beam.CombineFn)])):
  """MetricComputation represents one or more metric computations.

  The preprocessor is called with a PCollection of extracts (or list of extracts
  if query_key is used) to compute the initial combiner input state which is
  then passed to the combiner. This needs to be done in two steps because
  slicing happens between the call to the preprocessor and the combiner and this
  state may end up in multiple slices so we want the representation to be as
  efficient as possible. If the preprocessor is None, then StandardMetricInputs
  will be passed.

  A MetricComputation is uniquely identified by the combination of the
  combiner's name and the keys. Duplicate computations will be removed
  automatically.

  Attributes:
    keys: List of metric keys associated with computation. If the keys are
      defined as part of the computation then this may be empty in which case
      only the combiner name will be used for identifying computation
      uniqueness.
    preprocessor: Takes a extracts (or a list of extracts) as input (which
      typically will contain labels, predictions, example weights, and
      optionally features) and should return the initial state that the combiner
      will use as input. The output of a processor should only contain
      information needed by the combiner. Note that if a query_key is used the
      preprocessor will be passed a list of extracts as input representing the
      extracts that matched the query_key.  The special FeaturePreprocessor can
      be used to add additional features to the default standard metric inputs.
    combiner: Takes preprocessor output as input and outputs a tuple: (slice,
      metric results). The metric results should be a dict from MetricKey to
      value (float, int, distribution, ...).
  """

  def __new__(cls, keys: List[MetricKey], preprocessor: beam.DoFn,
              combiner: beam.CombineFn):
    return super(MetricComputation, cls).__new__(cls, keys, preprocessor,
                                                 combiner)

  def _computation_id(self):
    # Some computations do not define the keys until the end of the computation
    # is complete. In these cases the keys will be empty so we also distinguish
    # based on the combiner name used. We don't use __class__ since classes may
    # be defined inline which wouldn't compare equal.
    return (self.combiner.__class__.__name__, tuple(sorted(self.keys or [])))

  def __eq__(self, other):
    if isinstance(other, MetricComputation):
      return self._computation_id() == other._computation_id()
    else:
      return False

  def __hash__(self):
    return hash(self._computation_id())


class DerivedMetricComputation(
    NamedTuple(
        'DerivedMetricComputation',
        [('keys', List[MetricKey]),
         ('result', Callable)]  # Dict[MetricKey,Any] -> Dict[MetricKey,Any]
    )):
  """DerivedMetricComputation derives its result from other computations.

  When creating derived metric computations it is recommended (but not required)
  that the underlying MetricComputations that they depend on are defined at the
  same time. This is to avoid having to pre-construct and pass around all the
  required dependencies in order to construct a derived metric. The evaluation
  pipeline is responsible for de-duplicating overlapping MetricComputations so
  that only one computation is actually run.

  A DerivedMetricComputation is uniquely identified by the combination of the
  result function's name and the keys. Duplicate computations will be removed
  automatically.

  Attributes:
    keys: List of metric keys associated with derived computation. If the keys
      are defined as part of the computation then this may be empty in which
      case only the result function name will be used for identifying
      computation uniqueness.
    result: Function (called per slice) to compute the result using the results
      of other metric computations.
  """

  def __new__(cls, keys: List[MetricKey],
              result: Callable[[Dict[MetricKey, Any]], Dict[MetricKey, Any]]):
    return super(DerivedMetricComputation, cls).__new__(cls, keys, result)

  def _computation_id(self):
    # Some computations do not define the keys until the end of the computation
    # is complete. In these cases the keys will be empty so we also distinguish
    # based on the result function name used. We don't use __class__ since
    # functions may be defined inline which wouldn't compare equal.
    return (self.result.__class__.__name__, tuple(sorted(self.keys or [])))

  def __eq__(self, other):
    if isinstance(other, DerivedMetricComputation):
      return self._computation_id() == other._computation_id()
    else:
      return False

  def __hash__(self):
    return hash(self._computation_id())


CrossSliceComparisonCallable = Callable[
    [Dict[MetricKey, Any], Dict[MetricKey, Any]], Dict[MetricKey, Any]]


class CrossSliceMetricComputation(
    NamedTuple('CrossSliceMetricComputation', [
        ('keys', List[MetricKey]),
        ('cross_slice_comparison', CrossSliceComparisonCallable),
    ])):
  """CrossSliceMetricComputation derives its result from other computations.

  It is used for metrics which are based upon cross slice comparison.
  When creating these metric computations it is recommended (but not required)
  that the underlying MetricComputations that they depend on are defined at the
  same time. This is to avoid having to pre-construct and pass around all the
  required dependencies in order to construct a derived metric. The evaluation
  pipeline is responsible for de-duplicating overlapping MetricComputations so
  that only one computation is actually run.

  A CrossSliceMetricComputation is uniquely identified by the combination of the
  result function's name and the keys. Duplicate computations will be removed
  automatically.

  Attributes:
    keys: List of metric keys associated with derived computation. If the keys
      are defined as part of the computation then this may be empty in which
      case only the result function name will be used for identifying
      computation uniqueness.
    cross_slice_comparison: Function called to perform cross slice comparison
      using the results of the other metric computations.
  """

  def __new__(cls, keys: List[MetricKey],
              cross_slice_comparison: CrossSliceComparisonCallable):
    return super(CrossSliceMetricComputation,
                 cls).__new__(cls, keys, cross_slice_comparison)

  def _computation_id(self):
    # Some computations do not define the keys until the end of the computation
    # is complete. In these cases the keys will be empty so we also distinguish
    # based on the result function name used. We don't use __class__ since
    # functions may be defined inline which wouldn't compare equal.
    return (self.cross_slice_comparison.__class__.__name__,
            tuple(sorted(self.keys or [])))

  def __eq__(self, other):
    if isinstance(other, CrossSliceMetricComputation):
      return self._computation_id() == other._computation_id()
    else:
      return False

  def __hash__(self):
    return hash(self._computation_id())


# MetricComputations is a list of derived and non-derived computations used to
# calculate one or more metric values. Derived metrics should come after the
# computations they depend on in the list.
MetricComputations = List[Union[MetricComputation, DerivedMetricComputation,
                                CrossSliceMetricComputation]]


def update_create_computations_fn_kwargs(
    arg_names: Iterable[Text],
    kwargs: Dict[Text, Any],
    eval_config: Optional[config.EvalConfig] = None,
    schema: Optional[schema_pb2.Schema] = None,
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    sub_keys: Optional[List[Optional[SubKey]]] = None,
    aggregation_type: Optional[AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    query_key: Optional[Text] = None,
    is_diff: Optional[bool] = False):
  """Updates create_computations_fn kwargs based on arg spec.

  Each metric's create_computations_fn is invoked with a variable set of
  parameters, depending on the argument names of the callable. If an argument
  name matches one of the reserved names, this function will update the kwargs
  with the appropriate value for that arg.

  Args:
    arg_names: The arg_names for the create_computations_fn.
    kwargs: The existing kwargs for create_computations_fn.
    eval_config: The value to use when `eval_config` is in arg_names.
    schema: The value to use when `schema` is in arg_names.
    model_names: The value to use when `model_names` is in arg_names.
    output_names: The value to use when `output_names` is in arg_names.
    sub_keys: The value to use when `sub_keys` is in arg_names.
    aggregation_type: The value to use when `aggregation_type` is in arg_names.
    class_weights: The value to use when `class_weights` is in arg_names.
    query_key: The value to use when `query_key` is in arg_names.
    is_diff: The value to use when `is_diff` is in arg_names.

  Returns:
    The kwargs passed as input, updated with the appropriate additional args.
  """
  if 'eval_config' in arg_names:
    kwargs['eval_config'] = eval_config
  if 'schema' in arg_names:
    kwargs['schema'] = schema
  if 'model_names' in arg_names:
    kwargs['model_names'] = model_names
  if 'output_names' in arg_names:
    kwargs['output_names'] = output_names
  if 'sub_keys' in arg_names:
    kwargs['sub_keys'] = sub_keys
  if 'aggregation_type' in arg_names:
    kwargs['aggregation_type'] = aggregation_type
  if 'class_weights' in arg_names:
    kwargs['class_weights'] = class_weights
  if 'query_key' in arg_names:
    kwargs['query_key'] = query_key
  if 'is_diff' in arg_names:
    kwargs['is_diff'] = is_diff
  return kwargs


class Metric(object):
  """Metric wraps a set of metric computations.

  This class exists to provide similarity between tfma.metrics.Metric and
  tf.keras.metics.Metric.

  Calling computations creates the metric computations. The parameters passed to
  __init__ will be combined with the parameters passed to the computations
  method. This allows some of the parameters (e.g. model_names, output_names,
  sub_keys) to be set at the time the computations are created instead of when
  the metric is defined.
  """

  def __init__(self, create_computations_fn: Callable[..., MetricComputations],
               **kwargs):
    """Initializes metric.

    Args:
      create_computations_fn: Function to create the metrics computations (e.g.
        mean_label, etc). This function should take the args passed to __init__
        as as input along with any of eval_config, schema, model_names,
        output_names, sub_keys, aggregation_type, or query_key (where needed).
      **kwargs: Any additional kwargs to pass to create_computations_fn. These
        should only contain primitive types or lists/dicts of primitive types.
        The kwargs passed to computations have precendence over these kwargs.
    """
    self.create_computations_fn = create_computations_fn
    self.kwargs = kwargs
    if 'name' in kwargs:
      self.name = kwargs['name']
    else:
      self.name = None
    if hasattr(inspect, 'getfullargspec'):
      self._args = inspect.getfullargspec(self.create_computations_fn).args
    else:
      self._args = inspect.getargspec(self.create_computations_fn).args  # pylint: disable=deprecated-method

  def get_config(self) -> Dict[Text, Any]:
    """Returns serializable config."""
    return self.kwargs

  @property
  def compute_confidence_interval(self) -> bool:
    """Whether to compute confidence intervals for this metric.

    Note that this may not completely remove the computational overhead
    involved in computing a given metric. This is only respected by the
    jackknife confidence interval method.

    Returns:
      Whether to compute confidence intervals for this metric.
    """
    return True

  def computations(self,
                   eval_config: Optional[config.EvalConfig] = None,
                   schema: Optional[schema_pb2.Schema] = None,
                   model_names: Optional[List[Text]] = None,
                   output_names: Optional[List[Text]] = None,
                   sub_keys: Optional[List[Optional[SubKey]]] = None,
                   aggregation_type: Optional[AggregationType] = None,
                   class_weights: Optional[Dict[int, float]] = None,
                   query_key: Optional[Text] = None,
                   is_diff: Optional[bool] = False) -> MetricComputations:
    """Creates computations associated with metric."""
    updated_kwargs = update_create_computations_fn_kwargs(
        self._args, self.kwargs.copy(), eval_config, schema, model_names,
        output_names, sub_keys, aggregation_type, class_weights, query_key,
        is_diff)
    return self.create_computations_fn(**updated_kwargs)


_METRIC_OBJECTS = {}


def register_metric(cls: Type[Metric]):
  """Registers metric under the list of standard TFMA metrics."""
  _METRIC_OBJECTS[cls.__name__] = cls


def registered_metrics() -> Dict[Text, Type[Metric]]:
  """Returns standard TFMA metrics."""
  return copy.copy(_METRIC_OBJECTS)


class StandardMetricInputs(util.StandardExtracts):
  """Standard inputs used by most metric computations.

  StandardMetricInputs is a wrapper around Extracts where only the extracts keys
  used by one or more ExtractsPreprocessors will be present.
  """

  @property
  def label(self) -> Optional[types.TensorValueMaybeMultiLevelDict]:
    """Same as labels (DEPRECATED - use labels)."""
    return self.get_labels()

  @property
  def prediction(self) -> Optional[types.TensorValueMaybeMultiLevelDict]:
    """Same as predictions (DEPRECATED - use predictions)."""
    return self.get_predictions()

  @property
  def example_weight(self) -> Optional[types.TensorValueMaybeMultiLevelDict]:
    """Same as example_weights (DEPRECATED - use example_weights)."""
    return self.get_example_weights()

  def get_by_key(self,
                 key: Text,
                 model_name: Optional[Text] = None,
                 output_name: Optional[Text] = None) -> Any:
    if key not in self and key.endswith('s'):
      # The previous version of StandardMetricInputs was a NamedTuple that
      # used label, prediction, and example_weight as the field names. Some
      # tests may be creating StandardMetricInputs using these names, so also
      # search under the non-pluralized form of the key.
      key = key[:-1]
    return super(StandardMetricInputs, self).get_by_key(key, model_name,
                                                        output_name)


class StandardMetricInputsPreprocessor(beam.DoFn):
  """Preprocessor for filtering the extracts used in StandardMetricInputs."""

  def __init__(self,
               include_filter: Optional[Union[Iterable[Text],
                                              Dict[Text, Any]]] = None,
               include_default_inputs: bool = True,
               model_names: Optional[Iterable[Text]] = None,
               output_names: Optional[Iterable[Text]] = None):
    """Initializes preprocessor.

    Args:
      include_filter: Optional list or map of extracts keys to include in
        output. If a map of keys is passed then the keys and sub-keys that exist
        in the map will be included in the output. An empty dict behaves as a
        wildcard matching all keys or the value itself. Since matching on values
        is not currently supported, an empty dict must be used to represent the
        leaf nodes. For example, {'key1': {'key1-subkey': {}}, 'key2': {}}.
      include_default_inputs: True to include default inputs (labels,
        predictions, example weights) in addition to any inputs that may be
        specified using include_filter.
      model_names: Optional model names. Only used if include_default_inputs is
        True. If unset all models will be included with the default inputs.
      output_names: Optional output names. Only used if include_default_inputs
        is True. If unset all outputs will be included with the default inputs.
    """
    if include_filter is None:
      include_filter = {}
    if not isinstance(include_filter, MutableMapping):
      if isinstance(include_filter, Iterable):
        include_filter = {k: {} for k in include_filter or []}
      else:
        raise ValueError('include_filter must be a list or dict')

    if include_default_inputs:
      default_filter = {}
      if output_names:
        default_filter = {name: default_filter for name in output_names}
      if model_names:
        default_filter = {name: default_filter for name in model_names}
      include_filter = copy.copy(include_filter)
      include_filter.update({
          constants.LABELS_KEY: default_filter,
          constants.PREDICTIONS_KEY: default_filter,
          constants.EXAMPLE_WEIGHTS_KEY: default_filter,
      })
    self.include_filter = include_filter

  def process(self, extracts: types.Extracts) -> Iterable[types.Extracts]:
    if not self.include_filter:
      return {}
    yield util.include_filter(self.include_filter, extracts)


def InputPreprocessor(  # pylint: disable=invalid-name
    include_default_inputs: bool = False) -> StandardMetricInputsPreprocessor:
  """Returns preprocessor for including raw inputs in StandardMetricInputs.

  Args:
    include_default_inputs: True to include default inputs (labels, predictions,
      example weights) in addition to the inputs.
  """
  return StandardMetricInputsPreprocessor(
      include_filter={constants.INPUT_KEY: {}},
      include_default_inputs=include_default_inputs)


def FeaturePreprocessor(  # pylint: disable=invalid-name
    feature_keys: Iterable[Text],
    include_default_inputs: bool = True,
    model_names: Optional[Iterable[Text]] = None,
    output_names: Optional[Iterable[Text]] = None
) -> StandardMetricInputsPreprocessor:
  """Returns preprocessor for including features in StandardMetricInputs.

  Args:
    feature_keys: List of feature keys. An empty list means all.
    include_default_inputs: True to include default inputs (labels, predictions,
      example weights) in addition to the features.
    model_names: Optional model names. Only used if include_default_inputs is
      True. If unset all models will be included with the default inputs.
    output_names: Optional output names. Only used if include_default_inputs is
      True. If unset all outputs will be included with the default inputs.
  """
  if feature_keys:
    include_features = {k: {} for k in feature_keys}
  else:
    include_features = {}
  return StandardMetricInputsPreprocessor(
      include_filter={constants.FEATURES_KEY: include_features},
      include_default_inputs=include_default_inputs,
      model_names=model_names,
      output_names=output_names)


def TransformedFeaturePreprocessor(  # pylint: disable=invalid-name
    feature_keys: Iterable[Text],
    include_default_inputs: bool = True,
    model_names: Optional[Iterable[Text]] = None,
    output_names: Optional[Iterable[Text]] = None,
) -> StandardMetricInputsPreprocessor:
  """Returns preprocessor for incl transformed features in StandardMetricInputs.

  Args:
    feature_keys: List of feature keys. An empty list means all.
    include_default_inputs: True to include default inputs (labels, predictions,
      example weights) in addition to the transformed features.
    model_names: Optional model names (required if transformed_features used
      with multi-model evaluations).
    output_names: Optional output names. Only used if include_default_inputs is
      True. If unset all outputs will be included with the default inputs.
  """
  if feature_keys:
    include_features = {k: {} for k in feature_keys}
  else:
    include_features = {}
  if model_names:
    include_features = {name: include_features for name in model_names}
  return StandardMetricInputsPreprocessor(
      include_filter={constants.TRANSFORMED_FEATURES_KEY: include_features},
      include_default_inputs=include_default_inputs,
      model_names=model_names,
      output_names=output_names)


def AttributionPreprocessor(  # pylint: disable=invalid-name
    feature_keys: Iterable[Text],
    include_default_inputs: bool = True,
    model_names: Optional[Iterable[Text]] = None,
    output_names: Optional[Iterable[Text]] = None
) -> StandardMetricInputsPreprocessor:
  """Returns preprocessor for including attributions in StandardMetricInputs.

  Args:
    feature_keys: List of feature keys under attributions to keep. An empty list
      means all.
    include_default_inputs: True to include default inputs (labels, predictions,
      example weights) in addition to the transformed features.
    model_names: Optional model names (required for multi-model evaluations).
    output_names: Optional output names (required for multi-output evaluations).
  """
  if feature_keys:
    include_features = {k: {} for k in feature_keys}
  else:
    include_features = {}
  if output_names:
    include_features = {name: include_features for name in output_names}
  if model_names:
    include_features = {name: include_features for name in model_names}
  return StandardMetricInputsPreprocessor(
      include_filter={constants.ATTRIBUTIONS_KEY: include_features},
      include_default_inputs=include_default_inputs,
      model_names=model_names,
      output_names=output_names)


def StandardMetricInputsPreprocessorList(  # pylint: disable=invalid-name
    preprocessors: List[StandardMetricInputsPreprocessor]
) -> StandardMetricInputsPreprocessor:
  """Returns preprocessor combining multiple standard preprocessors together.

  Args:
    preprocessors: List of StandardMetricInputsPreprocessors. Must be of type
      StandardMetricInputsPreprocessor (subclasses not supported).
  """
  include_filter = {}
  for p in preprocessors:
    if type(p) != StandardMetricInputsPreprocessor:  # pylint: disable=unidiomatic-typecheck
      raise ValueError(
          'Only direct instances of StandardMetricsInputPreprocessor '
          '(excluding sub-classes) are supported')
    if not include_filter:
      include_filter = p.include_filter
    else:
      include_filter = util.merge_filters(include_filter, p.include_filter)
  return StandardMetricInputsPreprocessor(
      include_filter=include_filter, include_default_inputs=False)
