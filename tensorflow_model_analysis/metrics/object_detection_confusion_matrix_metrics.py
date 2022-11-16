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
"""Binary confusion matrices."""

from typing import List, Optional, Tuple, Union

from tensorflow_model_analysis.metrics import confusion_matrix_metrics
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import preprocessors
from tensorflow_model_analysis.proto import config_pb2

# default values for object detection settings
_DEFAULT_IOU_THRESHOLD = 0.5
_DEFAULT_AREA_RANGE = (0, float('inf'))

OBJECT_DETECTION_MAX_RECALL_NAME = 'object_detection_max_recall'
OBJECT_DETECTION_PRECISION_AT_RECALL_NAME = 'object_detection_precision_at_recall'
OBJECT_DETECTION_RECALL_NAME = 'object_detection_recall'
OBJECT_DETECTION_PRECISION_NAME = 'object_detection_precision'
OBJECT_DETECTION_THRESHOLD_AT_RECALL_NAME = 'object_detection_threshold_at_recall'


class ObjectDetectionPrecisionAtRecall(
    confusion_matrix_metrics.PrecisionAtRecall):
  """Computes best precision where recall is >= specified value.

  The threshold for the given recall value is computed and used to evaluate the
  corresponding precision.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               recall: Union[float, List[float]],
               thresholds: Optional[List[float]] = None,
               num_thresholds: Optional[int] = None,
               name: Optional[str] = None,
               iou_threshold: Optional[float] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes PrecisionAtRecall metric.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      recall: A scalar or a list of scalar values in range `[0, 1]`.
      thresholds: (Optional) Thresholds to use for calculating the matrices. Use
        one of either thresholds or num_thresholds.
      num_thresholds: (Optional) Defaults to 1000. The number of thresholds to
        use for matching the given recall.
      name: (Optional) string name of the metric instance.
      iou_threshold: (Optional) Thresholds for a detection and ground truth pair
        with specific iou to be considered as a match. Default to 0.5
      class_id: (Optional) The class id for calculating metrics.
      class_weight: (Optional) The weight associated with the object class id.
      area_range: (Optional) A tuple (inclusive) representing the area-range for
        objects to be considered for metrics. Default to (0, inf).
      max_num_detections: (Optional) The maximum number of detections for a
        single image. Default to None.
      labels_to_stack: (Optional) Keys for columns to be stacked as a single
        numpy array as the labels. It is searched under the key labels, features
        and transformed features. The desired format is [left bounadary, top
        boudnary, right boundary, bottom boundary, class id]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id']
      predictions_to_stack: (Optional) Output names for columns to be stacked as
        a single numpy array as the prediction. It should be the model's output
        names. The desired format is [left bounadary, top boudnary, right
        boundary, bottom boundary, class id, confidence score]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id', 'scores']
      num_detections_key: (Optional) An output name in which to find the number
        of detections to use for evaluation for a given example. It does nothing
        if predictions_to_stack is not set. The value for this output should be
        a scalar value or a single-value tensor. The stacked predicitions will
        be truncated with the specified number of detections.
    """
    for r in [recall] if isinstance(recall, float) else recall:
      if r < 0 or r > 1:
        raise ValueError('Argument `recall` must be in the range [0, 1]. '
                         f'Received: recall={r}')

    super().__init__(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        recall=recall,
        name=name,
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return OBJECT_DETECTION_PRECISION_AT_RECALL_NAME

  def _metric_computations(self,
                           thresholds: Optional[Union[float,
                                                      List[float]]] = None,
                           num_thresholds: Optional[int] = None,
                           iou_threshold: Optional[float] = None,
                           class_id: Optional[int] = None,
                           class_weight: Optional[float] = None,
                           area_range: Optional[Tuple[float, float]] = None,
                           max_num_detections: Optional[int] = None,
                           name: Optional[str] = None,
                           eval_config: Optional[config_pb2.EvalConfig] = None,
                           model_name: str = '',
                           output_name: str = '',
                           labels_to_stack: Optional[List[str]] = None,
                           predictions_to_stack: Optional[List[str]] = None,
                           num_detections_key: Optional[str] = None,
                           **kwargs) -> metric_types.MetricComputations:
    metric_util.validate_object_detection_arguments(
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    preprocessor = preprocessors.BoundingBoxMatchPreprocessor(
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key,
        model_name=model_name)
    return super()._metric_computations(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        preprocessors=[preprocessor],
        **kwargs)


metric_types.register_metric(ObjectDetectionPrecisionAtRecall)


class ObjectDetectionRecall(confusion_matrix_metrics.Recall):
  """Computes the recall of the predictions with respect to the labels.

  The metric uses true positives and false negatives to compute recall by
  dividing the true positives by the sum of true positives and false negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               iou_threshold: Optional[float] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes Recall metric.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. The default is to calculate recall with
        `thresholds=0.5`.
      name: (Optional) string name of the metric instance.
      iou_threshold: (Optional) Thresholds for a detection and ground truth pair
        with specific iou to be considered as a match. Default to 0.5
      class_id: (Optional) The class id for calculating metrics.
      class_weight: (Optional) The weight associated with the object class id.
      area_range: (Optional) A tuple (inclusive) representing the area-range for
        objects to be considered for metrics. Default to (0, inf).
      max_num_detections: (Optional) The maximum number of detections for a
        single image. Default to None.
      labels_to_stack: (Optional) Keys for columns to be stacked as a single
        numpy array as the labels. It is searched under the key labels, features
        and transformed features. The desired format is [left bounadary, top
        boudnary, right boundary, bottom boundary, class id]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id']
      predictions_to_stack: (Optional) Output names for columns to be stacked as
        a single numpy array as the prediction. It should be the model's output
        names. The desired format is [left bounadary, top boudnary, right
        boundary, bottom boundary, class id, confidence score]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id', 'scores']
      num_detections_key: (Optional) An output name in which to find the number
        of detections to use for evaluation for a given example. It does nothing
        if predictions_to_stack is not set. The value for this output should be
        a scalar value or a single-value tensor. The stacked predicitions will
        be truncated with the specified number of detections.
    """
    super().__init__(
        thresholds=thresholds,
        name=name,
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return OBJECT_DETECTION_RECALL_NAME

  def _metric_computations(
      self,
      thresholds: Optional[Union[float, List[float]]] = None,
      num_thresholds: Optional[int] = None,
      iou_threshold: Optional[float] = None,
      class_id: Optional[int] = None,
      class_weight: Optional[float] = None,
      area_range: Optional[Tuple[float, float]] = None,
      max_num_detections: Optional[int] = None,
      name: Optional[str] = None,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      output_name: str = '',
      labels_to_stack: Optional[List[str]] = None,
      predictions_to_stack: Optional[List[str]] = None,
      num_detections_key: Optional[str] = None,
      **kwargs,
  ) -> metric_types.MetricComputations:
    metric_util.validate_object_detection_arguments(
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    preprocessor = preprocessors.BoundingBoxMatchPreprocessor(
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key,
        model_name=model_name)
    return super()._metric_computations(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        preprocessors=[preprocessor])


metric_types.register_metric(ObjectDetectionRecall)


class ObjectDetectionPrecision(confusion_matrix_metrics.Precision):
  """Computes the precision of the predictions with respect to the labels.

  The metric uses true positives and false positives to compute precision by
  dividing the true positives by the sum of true positives and false positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               iou_threshold: Optional[float] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes Recall metric.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. The default is to calculate precision with
        `thresholds=0.5`.
      name: (Optional) string name of the metric instance.
      iou_threshold: (Optional) Thresholds for a detection and ground truth pair
        with specific iou to be considered as a match. Default to 0.5
      class_id: (Optional) The class id for calculating metrics.
      class_weight: (Optional) The weight associated with the object class id.
      area_range: (Optional) A tuple (inclusive) representing the area-range for
        objects to be considered for metrics. Default to (0, inf).
      max_num_detections: (Optional) The maximum number of detections for a
        single image. Default to None.
      labels_to_stack: (Optional) Keys for columns to be stacked as a single
        numpy array as the labels. It is searched under the key labels, features
        and transformed features. The desired format is [left bounadary, top
        boudnary, right boundary, bottom boundary, class id]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id']
      predictions_to_stack: (Optional) Output names for columns to be stacked as
        a single numpy array as the prediction. It should be the model's output
        names. The desired format is [left bounadary, top boudnary, right
        boundary, bottom boundary, class id, confidence score]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id', 'scores']
      num_detections_key: (Optional) An output name in which to find the number
        of detections to use for evaluation for a given example. It does nothing
        if predictions_to_stack is not set. The value for this output should be
        a scalar value or a single-value tensor. The stacked predicitions will
        be truncated with the specified number of detections.
    """
    super().__init__(
        thresholds=thresholds,
        name=name,
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return OBJECT_DETECTION_PRECISION_NAME

  def _metric_computations(
      self,
      thresholds: Optional[Union[float, List[float]]] = None,
      num_thresholds: Optional[int] = None,
      iou_threshold: Optional[float] = None,
      class_id: Optional[int] = None,
      class_weight: Optional[float] = None,
      area_range: Optional[Tuple[float, float]] = None,
      max_num_detections: Optional[int] = None,
      name: Optional[str] = None,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      output_name: str = '',
      labels_to_stack: Optional[List[str]] = None,
      predictions_to_stack: Optional[List[str]] = None,
      num_detections_key: Optional[str] = None,
      **kwargs,
  ) -> metric_types.MetricComputations:
    metric_util.validate_object_detection_arguments(
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    preprocessor = preprocessors.BoundingBoxMatchPreprocessor(
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key,
        model_name=model_name)
    return super()._metric_computations(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        preprocessors=[preprocessor])


metric_types.register_metric(ObjectDetectionPrecision)


class ObjectDetectionMaxRecall(confusion_matrix_metrics.MaxRecall):
  """Computes the max recall of the predictions with respect to the labels.

  The metric uses true positives and false negatives to compute recall by
  dividing the true positives by the sum of true positives and false negatives.

  Effectively the recall at threshold = epsilon(1.0e-12). It is equilvalent
  to the recall defined in COCO metrics.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               name: Optional[str] = None,
               iou_threshold: Optional[float] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes MaxRecall metrics, it calculates the maximum recall.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      name: (Optional) string name of the metric instance.
      iou_threshold: (Optional) Thresholds for a detection and ground truth pair
        with specific iou to be considered as a match. Default to 0.5
      class_id: (Optional) The class id for calculating metrics.
      class_weight: (Optional) The weight associated with the object class id.
      area_range: (Optional) A tuple (inclusive) representing the area-range for
        objects to be considered for metrics. Default to (0, inf).
      max_num_detections: (Optional) The maximum number of detections for a
        single image. Default to None.
      labels_to_stack: (Optional) Keys for columns to be stacked as a single
        numpy array as the labels. It is searched under the key labels, features
        and transformed features. The desired format is [left bounadary, top
        boudnary, right boundary, bottom boundary, class id]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id']
      predictions_to_stack: (Optional) Output names for columns to be stacked as
        a single numpy array as the prediction. It should be the model's output
        names. The desired format is [left bounadary, top boudnary, right
        boundary, bottom boundary, class id, confidence score]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id', 'scores']
      num_detections_key: (Optional) An output name in which to find the number
        of detections to use for evaluation for a given example. It does nothing
        if predictions_to_stack is not set. The value for this output should be
        a scalar value or a single-value tensor. The stacked predicitions will
        be truncated with the specified number of detections.
    """
    super().__init__(
        name=name,
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return OBJECT_DETECTION_MAX_RECALL_NAME

  def _metric_computations(self,
                           thresholds: Optional[Union[float,
                                                      List[float]]] = None,
                           num_thresholds: Optional[int] = None,
                           iou_threshold: Optional[float] = None,
                           class_id: Optional[int] = None,
                           class_weight: Optional[float] = None,
                           area_range: Optional[Tuple[float, float]] = None,
                           max_num_detections: Optional[int] = None,
                           name: Optional[str] = None,
                           eval_config: Optional[config_pb2.EvalConfig] = None,
                           model_name: str = '',
                           output_name: str = '',
                           labels_to_stack: Optional[List[str]] = None,
                           predictions_to_stack: Optional[List[str]] = None,
                           num_detections_key: Optional[str] = None,
                           **kwargs) -> metric_types.MetricComputations:
    metric_util.validate_object_detection_arguments(
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack)
    preprocessor = preprocessors.BoundingBoxMatchPreprocessor(
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key,
        model_name=model_name)
    return super()._metric_computations(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        preprocessors=[preprocessor],
        **kwargs)


metric_types.register_metric(ObjectDetectionMaxRecall)


class ObjectDetectionThresholdAtRecall(
    confusion_matrix_metrics.ThresholdAtRecall):
  """Computes maximum threshold where recall is >= specified value.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               recall: Union[float, List[float]],
               thresholds: Optional[List[float]] = None,
               num_thresholds: Optional[int] = None,
               name: Optional[str] = None,
               iou_threshold: Optional[float] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes ThresholdAtRecall metric.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      recall: A scalar or a list of scalar values in range `[0, 1]`.
      thresholds: (Optional) Thresholds to use for calculating the matrices. Use
        one of either thresholds or num_thresholds.
      num_thresholds: (Optional) Defaults to 1000. The number of thresholds to
        use for matching the given recall.
      name: (Optional) string name of the metric instance.
      iou_threshold: (Optional) Thresholds for a detection and ground truth pair
        with specific iou to be considered as a match. Default to 0.5
      class_id: (Optional) The class id for calculating metrics.
      class_weight: (Optional) The weight associated with the object class id.
      area_range: (Optional) A tuple (inclusive) representing the area-range for
        objects to be considered for metrics. Default to (0, inf).
      max_num_detections: (Optional) The maximum number of detections for a
        single image. Default to None.
      labels_to_stack: (Optional) Keys for columns to be stacked as a single
        numpy array as the labels. It is searched under the key labels, features
        and transformed features. The desired format is [left bounadary, top
        boudnary, right boundary, bottom boundary, class id]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id']
      predictions_to_stack: (Optional) Output names for columns to be stacked as
        a single numpy array as the prediction. It should be the model's output
        names. The desired format is [left bounadary, top boudnary, right
        boundary, bottom boundary, class id, confidence score]. e.g. ['xmin',
        'ymin', 'xmax', 'ymax', 'class_id', 'scores']
      num_detections_key: (Optional) An output name in which to find the number
        of detections to use for evaluation for a given example. It does nothing
        if predictions_to_stack is not set. The value for this output should be
        a scalar value or a single-value tensor. The stacked predicitions will
        be truncated with the specified number of detections.
    """
    for r in [recall] if isinstance(recall, float) else recall:
      if r < 0 or r > 1:
        raise ValueError('Argument `recall` must be in the range [0, 1]. '
                         f'Received: recall={r}')

    super().__init__(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        recall=recall,
        name=name,
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return OBJECT_DETECTION_THRESHOLD_AT_RECALL_NAME

  def _metric_computations(self,
                           thresholds: Optional[Union[float,
                                                      List[float]]] = None,
                           num_thresholds: Optional[int] = None,
                           iou_threshold: Optional[float] = None,
                           class_id: Optional[int] = None,
                           class_weight: Optional[float] = None,
                           area_range: Optional[Tuple[float, float]] = None,
                           max_num_detections: Optional[int] = None,
                           name: Optional[str] = None,
                           eval_config: Optional[config_pb2.EvalConfig] = None,
                           model_name: str = '',
                           output_name: str = '',
                           labels_to_stack: Optional[List[str]] = None,
                           predictions_to_stack: Optional[List[str]] = None,
                           num_detections_key: Optional[str] = None,
                           **kwargs) -> metric_types.MetricComputations:
    metric_util.validate_object_detection_arguments(
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    preprocessor = preprocessors.BoundingBoxMatchPreprocessor(
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key,
        model_name=model_name)
    return super()._metric_computations(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        preprocessors=[preprocessor],
        **kwargs)


metric_types.register_metric(ObjectDetectionThresholdAtRecall)
