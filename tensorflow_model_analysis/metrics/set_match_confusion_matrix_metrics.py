# Copyright 2023 Google LLC
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
"""set match confusion matrices."""

from typing import List, Optional, Union, Dict

from tensorflow_model_analysis.metrics import confusion_matrix_metrics
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import preprocessors
from tensorflow_model_analysis.proto import config_pb2

SET_MATCH_PRECISION_NAME = 'set_match_precision'
SET_MATCH_RECALL_NAME = 'set_match_recall'


class SetMatchPrecision(confusion_matrix_metrics.Precision):
  """Computes precision for sets of labels and predictions.

  The metric deals with labels and predictions which are provided in the format
  of sets (stored as variable length numpy arrays). The precision is the
  micro averaged classification precision. The metric is suitable for the case
  where the number of classes is large or the list of classes could not be
  provided in advance.

  Example:
  Label: ['cats'],
  Predictions: {'classes': ['cats, dogs']}

  The precision is 0.5.
  """

  def __init__(
      self,
      thresholds: Optional[Union[float, List[float]]] = None,
      top_k: Optional[int] = None,
      name: Optional[str] = None,
      prediction_class_key: str = 'classes',
      prediction_score_key: str = 'scores',
      class_key: Optional[str] = None,
      weight_key: Optional[str] = None,
      **kwargs,
  ):
    """Initializes Precision metric.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. If neither thresholds nor top_k are set, the
        default is to calculate precision with `thresholds=0.5`.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are truncated and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. When
        top_k is used, the default threshold is float('-inf'). In this case,
        unmatched labels are still considered false negative, since they have
        prediction with confidence score float('-inf'),
      name: (Optional) string name of the metric instance.
      prediction_class_key: the key name of the classes in prediction.
      prediction_score_key: the key name of the scores in prediction.
      class_key: (Optional) The key name of the classes in class-weight pairs.
        If it is not provided, the classes are assumed to be the label classes.
      weight_key: (Optional) The key name of the weights of classes in
        class-weight pairs. The value in this key should be a numpy array of the
        same length as the classes in class_key. The key should be stored under
        the features key.
      **kwargs: (Optional) Additional args to pass along to init (and eventually
        on to _metric_computations and _metric_values). The args are passed to
        the precision metric, the confusion matrix metric and binary
        classification metric.
    """

    super().__init__(
        thresholds=thresholds,
        top_k=top_k,
        name=name,
        prediction_class_key=prediction_class_key,
        prediction_score_key=prediction_score_key,
        class_key=class_key,
        weight_key=weight_key,
        **kwargs,
    )

  def _default_name(self) -> str:
    return SET_MATCH_PRECISION_NAME

  def _metric_computations(
      self,
      thresholds: Optional[Union[float, List[float]]] = None,
      top_k: Optional[int] = None,
      name: Optional[str] = None,
      prediction_class_key: str = 'classes',
      prediction_score_key: str = 'scores',
      class_key: Optional[str] = None,
      weight_key: Optional[str] = None,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      sub_key: Optional[metric_types.SubKey] = None,
      aggregation_type: Optional[metric_types.AggregationType] = None,
      class_weights: Optional[Dict[int, float]] = None,
      example_weighted: bool = False,
      **kwargs,
  ) -> metric_types.MetricComputations:
    preprocessor = preprocessors.SetMatchPreprocessor(
        top_k=top_k,
        model_name=model_name,
        prediction_class_key=prediction_class_key,
        prediction_score_key=prediction_score_key,
        class_key=class_key,
        weight_key=weight_key,
    )
    if top_k is not None and thresholds is None:
      thresholds = float('-inf')

    if weight_key:
      # If example_weighted is False, it will by default set the example weights
      # to 1.0.
      # example_weighted could only be turned on from model_specs. However, in
      # this case, the example_weights is not provided in the models. It should
      # be turned on when per class weights are given.
      example_weighted = True
    return super()._metric_computations(
        thresholds=thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        preprocessors=[preprocessor],
        sub_key=sub_key,
        aggregation_type=aggregation_type,
        class_weights=class_weights,
        example_weighted=example_weighted,
        **kwargs,
    )


metric_types.register_metric(SetMatchPrecision)


class SetMatchRecall(confusion_matrix_metrics.Recall):
  """Computes recall for sets of labels and predictions.

  The metric deals with labels and predictions which are provided in the format
  of sets (stored as variable length numpy arrays). The recall is the
  micro averaged classification recall. The metric is suitable for the case
  where the number of classes is large or the list of classes could not be
  provided in advance.

  Example:
  Label: ['cats'],
  Predictions: {'classes': ['cats, dogs']}

  The recall is 1.
  """

  def __init__(
      self,
      thresholds: Optional[Union[float, List[float]]] = None,
      top_k: Optional[int] = None,
      name: Optional[str] = None,
      prediction_class_key: str = 'classes',
      prediction_score_key: str = 'scores',
      class_key: Optional[str] = None,
      weight_key: Optional[str] = None,
      **kwargs,
  ):
    """Initializes recall metric.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. If neither thresholds nor top_k are set, the
        default is to calculate precision with `thresholds=0.5`.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are truncated and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. When
        top_k is used, the default threshold is float('-inf'). In this case,
        unmatched labels are still considered false negative, since they have
        prediction with confidence score float('-inf'),
      name: (Optional) string name of the metric instance.
      prediction_class_key: the key name of the classes in prediction.
      prediction_score_key: the key name of the scores in prediction.
      class_key: (Optional) The key name of the classes in class-weight pairs.
        If it is not provided, the classes are assumed to be the label classes.
      weight_key: (Optional) The key name of the weights of classes in
        class-weight pairs. The value in this key should be a numpy array of the
        same length as the classes in class_key. The key should be stored under
        the features key.
      **kwargs: (Optional) Additional args to pass along to init (and eventually
        on to _metric_computations and _metric_values). The args are passed to
        the recall metric, the confusion matrix metric and binary classification
        metric.
    """

    super().__init__(
        thresholds=thresholds,
        top_k=top_k,
        name=name,
        prediction_class_key=prediction_class_key,
        prediction_score_key=prediction_score_key,
        class_key=class_key,
        weight_key=weight_key,
        **kwargs,
    )

  def _default_name(self) -> str:
    return SET_MATCH_RECALL_NAME

  def _metric_computations(
      self,
      thresholds: Optional[Union[float, List[float]]] = None,
      top_k: Optional[int] = None,
      name: Optional[str] = None,
      prediction_class_key: str = 'classes',
      prediction_score_key: str = 'scores',
      class_key: Optional[str] = None,
      weight_key: Optional[str] = None,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      sub_key: Optional[metric_types.SubKey] = None,
      aggregation_type: Optional[metric_types.AggregationType] = None,
      class_weights: Optional[Dict[int, float]] = None,
      example_weighted: bool = False,
      **kwargs,
  ) -> metric_types.MetricComputations:
    preprocessor = preprocessors.SetMatchPreprocessor(
        top_k=top_k,
        model_name=model_name,
        prediction_class_key=prediction_class_key,
        prediction_score_key=prediction_score_key,
        class_key=class_key,
        weight_key=weight_key,
    )
    if top_k is not None and thresholds is None:
      thresholds = float('-inf')
    if weight_key:
      # If example_weighted is False, it will by default set the example weights
      # to 1.0.
      # example_weighted could only be turned on from model_specs. However, in
      # this case, the example_weights is not provided in the models. It should
      # be turned on when per class weights are given.
      example_weighted = True
    return super()._metric_computations(
        thresholds=thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        preprocessors=[preprocessor],
        sub_key=sub_key,
        aggregation_type=aggregation_type,
        class_weights=class_weights,
        example_weighted=example_weighted,
        **kwargs,
    )


metric_types.register_metric(SetMatchRecall)
