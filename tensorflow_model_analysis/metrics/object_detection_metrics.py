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
"""COCO object detection metrics."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tensorflow_model_analysis.metrics import confusion_matrix_metrics
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import object_detection_confusion_matrix_metrics

AVERAGE_RECALL_NAME = 'average_recall'
AVERAGE_PRECISION_NAME = 'average_precision'
MEAN_AVERAGE_PRECISION_NAME = 'mean_average_precision'
MEAN_AVERAGE_RECALL_NAME = 'mean_average_recall'


class COCOAveragePrecision(metric_types.Metric):
  """Confusion matrix at thresholds.

  It computes the average precision of object detections for a single class and
  a single iou_threshold.
  """

  def __init__(self,
               num_thresholds: Optional[int] = None,
               iou_threshold: Optional[float] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               recalls: Optional[List[float]] = None,
               num_recalls: Optional[int] = None,
               name: Optional[str] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initialize average precision metric.

    This metric is only used in object-detection setting. It does not support
    sub_key parameters due to the matching algorithm of bounding boxes.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      num_thresholds: (Optional) Number of thresholds to use for calculating the
        matrices and finding the precision at given recall.
      iou_threshold: (Optional) Threholds for a detection and ground truth pair
        with specific iou to be considered as a match.
      class_id: (Optional) The class id for calculating metrics.
      class_weight: (Optional) The weight associated with the object class id.
      area_range: (Optional) The area-range for objects to be considered for
        metrics.
      max_num_detections: (Optional) The maximum number of detections for a
        single image.
      recalls: (Optional) recalls at which precisions will be calculated.
      num_recalls: (Optional) Used for objecth detection, the number of recalls
        for calculating average precision, it equally generates points bewteen 0
        and 1. (Only one of recalls and num_recalls should be used).
      name: (Optional) string name of the metric instance.
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
    if recalls is not None:
      recall_thresholds = recalls
    elif num_recalls is not None:
      recall_thresholds = np.linspace(0.0, 1.0, num_recalls)
    else:
      # by default set recall_thresholds to [0.0:0.01:1.0].
      recall_thresholds = np.linspace(0.0, 1.0, 101)

    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        num_thresholds=num_thresholds,
        iou_threshold=iou_threshold,
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        recall_thresholds=recall_thresholds,
        name=name,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return AVERAGE_PRECISION_NAME

  def _metric_computations(
      self,
      num_thresholds: Optional[int] = None,
      iou_threshold: Optional[float] = None,
      class_id: Optional[int] = None,
      class_weight: Optional[float] = None,
      max_num_detections: Optional[int] = None,
      area_range: Optional[Tuple[float, float]] = None,
      recall_thresholds: Optional[List[float]] = None,
      name: Optional[str] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False,
      labels_to_stack: Optional[List[str]] = None,
      predictions_to_stack: Optional[List[str]] = None,
      num_detections_key: Optional[str] = None
  ) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""
    metric_util.validate_object_detection_arguments(
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    if recall_thresholds is None:
      # If recall thresholds is not defined, initialize it as [0.0]
      recall_thresholds = [0.0]

    if num_thresholds is None:
      num_thresholds = 10000
    thresholds = [1.e-12] + [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ] + [1.0 - 1.e-12]
    # PrecisionAtRecall is a public function. To hide it from users who do not
    # need it, we make the name private with '_'.
    precision_at_recall_name = metric_util.generate_private_name_from_arguments(
        confusion_matrix_metrics.PRECISION_AT_RECALL_NAME,
        recall=recall_thresholds,
        num_thresholds=num_thresholds,
        iou_threshold=iou_threshold,
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections)

    pr = (
        object_detection_confusion_matrix_metrics
        .ObjectDetectionPrecisionAtRecall(
            recall=recall_thresholds,
            thresholds=thresholds,
            iou_threshold=iou_threshold,
            class_id=class_id,
            class_weight=class_weight,
            area_range=area_range,
            max_num_detections=max_num_detections,
            name=precision_at_recall_name,
            labels_to_stack=labels_to_stack,
            predictions_to_stack=predictions_to_stack,
            num_detections_key=num_detections_key))
    computations = pr.computations(
        model_names=[model_name], output_names=[output_name])
    precisions_key = computations[-1].keys[-1]

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      value = np.nanmean(metrics[precisions_key])
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOAveragePrecision)


class COCOMeanAveragePrecision(metric_types.Metric):
  """Mean average precision for object detections.

  It calculates the mean average precision metric for object detections. It
  averages COCOAveragePrecision over multiple classes and IoU thresholds.
  """

  def __init__(self,
               num_thresholds: Optional[int] = None,
               iou_thresholds: Optional[List[float]] = None,
               class_ids: Optional[List[int]] = None,
               class_weights: Optional[List[float]] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               recalls: Optional[List[float]] = None,
               num_recalls: Optional[int] = None,
               name: Optional[str] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes mean average precision metric.

    This metric is only used in object-detection setting. It does not support
    sub_key parameters due to the matching algorithm of bounding boxes.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      num_thresholds: (Optional) Number of thresholds to use for calculating the
        matrices and finding the precision at given recall.
      iou_thresholds: (Optional) Threholds for a detection and ground truth pair
        with specific iou to be considered as a match.
      class_ids: (Optional) The class ids for calculating metrics.
      class_weights: (Optional) The weight associated with the object class ids.
        If it is provided, it should have the same length as class_ids.
      area_range: (Optional) The area-range for objects to be considered for
        metrics.
      max_num_detections: (Optional) The maximum number of detections for a
        single image.
      recalls: (Optional) recalls at which precisions will be calculated.
      num_recalls: (Optional) Used for objecth detection, the number of recalls
        for calculating average precision, it equally generates points bewteen 0
        and 1. (Only one of recalls and num_recalls should be used).
      name: (Optional) Metric name.
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
        metric_util.merge_per_key_computations(self._metric_computations),
        num_thresholds=num_thresholds,
        iou_thresholds=iou_thresholds,
        class_ids=class_ids,
        class_weights=class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections,
        recalls=recalls,
        num_recalls=num_recalls,
        name=name,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return MEAN_AVERAGE_PRECISION_NAME

  def _metric_computations(self,
                           num_thresholds: Optional[int] = None,
                           iou_thresholds: Optional[List[float]] = None,
                           class_ids: Optional[List[int]] = None,
                           class_weights: Optional[List[float]] = None,
                           max_num_detections: Optional[int] = None,
                           area_range: Optional[Tuple[float, float]] = None,
                           recalls: Optional[List[float]] = None,
                           num_recalls: Optional[int] = None,
                           name: Optional[str] = None,
                           model_name: str = '',
                           output_name: str = '',
                           example_weighted: bool = False,
                           labels_to_stack: Optional[List[str]] = None,
                           predictions_to_stack: Optional[List[str]] = None,
                           num_detections_key: Optional[str] = None,
                           **kwargs) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""

    metric_util.validate_object_detection_arguments(
        class_id=class_ids,
        class_weight=class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    # set default value according to COCO metrics
    if iou_thresholds is None:
      iou_thresholds = np.linspace(0.5, 0.95, 10)
    if class_weights is None:
      class_weights = [1.0] * len(class_ids)

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    computations = []
    precisions_keys = []
    for iou_threshold in iou_thresholds:
      for class_id, class_weight in zip(class_ids, class_weights):

        average_precision_name = (
            metric_util.generate_private_name_from_arguments(
                AVERAGE_PRECISION_NAME,
                recall=recalls,
                num_recalls=num_recalls,
                num_thresholds=num_thresholds,
                iou_threshold=iou_threshold,
                class_id=class_id,
                class_weight=class_weight,
                area_range=area_range,
                max_num_detections=max_num_detections))

        ap = COCOAveragePrecision(
            num_thresholds=num_thresholds,
            iou_threshold=iou_threshold,
            class_id=class_id,
            class_weight=class_weight,
            area_range=area_range,
            max_num_detections=max_num_detections,
            recalls=recalls,
            num_recalls=num_recalls,
            name=average_precision_name,
            labels_to_stack=labels_to_stack,
            predictions_to_stack=predictions_to_stack,
            num_detections_key=num_detections_key)
        computations.extend(
            ap.computations(
                model_names=[model_name], output_names=[output_name]))
        precisions_keys.append(computations[-1].keys[-1])

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      precisions = [
          metrics[precisions_key] for precisions_key in precisions_keys
      ]
      value = np.nanmean(precisions)
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOMeanAveragePrecision)


class COCOAverageRecall(metric_types.Metric):
  """Average recall metric for object detection.

  It computes the average precision metric for object detections for a single
  class. It averages MaxRecall metric over mulitple IoU thresholds.
  """

  def __init__(self,
               iou_thresholds: Optional[List[float]] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               name: Optional[str] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes average recall metric.

    This metric is only used in object-detection setting. It does not support
    sub_key parameters due to the matching algorithm of bounding boxes.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      iou_thresholds: (Optional) Threholds for a detection and ground truth pair
        with specific iou to be considered as a match.
      class_id: (Optional) The class ids for calculating metrics.
      class_weight: (Optional) The weight associated with the object class ids.
        If it is provided, it should have the same length as class_ids.
      area_range: (Optional) The area-range for objects to be considered for
        metrics.
      max_num_detections: (Optional) The maximum number of detections for a
        single image.
      name: (Optional) Metric name.
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
        metric_util.merge_per_key_computations(self._metric_computations),
        iou_thresholds=iou_thresholds,
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        name=name,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return AVERAGE_RECALL_NAME

  def _metric_computations(
      self,
      iou_thresholds: Optional[Union[float, List[float]]] = None,
      class_id: Optional[int] = None,
      class_weight: Optional[float] = None,
      max_num_detections: Optional[int] = None,
      area_range: Optional[Tuple[float, float]] = None,
      name: Optional[str] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False,
      labels_to_stack: Optional[List[str]] = None,
      predictions_to_stack: Optional[List[str]] = None,
      num_detections_key: Optional[str] = None
  ) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""

    metric_util.validate_object_detection_arguments(
        class_id=class_id,
        class_weight=class_weight,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    # set default value according to COCO metrics
    if iou_thresholds is None:
      iou_thresholds = np.linspace(0.5, 0.95, 10)
    if class_weight is None:
      class_weight = 1.0

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    computations = []
    recalls_keys = []
    for iou_threshold in iou_thresholds:
      max_recall_name = metric_util.generate_private_name_from_arguments(
          confusion_matrix_metrics.MAX_RECALL_NAME,
          iou_threshold=iou_threshold,
          class_id=class_id,
          class_weight=class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections)

      mr = object_detection_confusion_matrix_metrics.ObjectDetectionMaxRecall(
          iou_threshold=iou_threshold,
          class_id=class_id,
          class_weight=class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections,
          name=max_recall_name,
          labels_to_stack=labels_to_stack,
          predictions_to_stack=predictions_to_stack,
          num_detections_key=num_detections_key)
      computations.extend(
          mr.computations(model_names=[model_name], output_names=[output_name]))
      recalls_keys.append(computations[-1].keys[-1])

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      for recalls_key in recalls_keys:
        if math.isnan(metrics[recalls_key]):
          logging.warning(
              'Recall with metric key %s is NaN, it will be'
              ' ignored in the following calculation.', recalls_key)
      recalls = [metrics[recalls_key] for recalls_key in recalls_keys]
      value = np.nanmean(recalls)
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOAverageRecall)


class COCOMeanAverageRecall(metric_types.Metric):
  """Mean Average recall metric for object detection.

  It computes the mean average precision metric for object detections for a
  single class. It averages COCOAverageRecall metric over mulitple classes.
  """

  def __init__(self,
               iou_thresholds: Optional[List[float]] = None,
               class_ids: Optional[List[int]] = None,
               class_weights: Optional[List[float]] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               name: Optional[str] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes average recall metric.

    This metric is only used in object-detection setting. It does not support
    sub_key parameters due to the matching algorithm of bounding boxes.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      iou_thresholds: (Optional) Threholds for a detection and ground truth pair
        with specific iou to be considered as a match.
      class_ids: (Optional) The class ids for calculating metrics.
      class_weights: (Optional) The weight associated with the object class ids.
        If it is provided, it should have the same length as class_ids.
      area_range: (Optional) The area-range for objects to be considered for
        metrics.
      max_num_detections: (Optional) The maximum number of detections for a
        single image.
      name: (Optional) Metric name.
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
        metric_util.merge_per_key_computations(self._metric_computations),
        iou_thresholds=iou_thresholds,
        class_ids=class_ids,
        class_weights=class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections,
        name=name,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _default_name(self) -> str:
    return MEAN_AVERAGE_RECALL_NAME

  def _metric_computations(
      self,
      iou_thresholds: Optional[List[float]] = None,
      class_ids: Optional[Union[int, List[int]]] = None,
      class_weights: Optional[Union[float, List[float]]] = None,
      max_num_detections: Optional[int] = None,
      area_range: Optional[Tuple[float, float]] = None,
      name: Optional[str] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False,
      labels_to_stack: Optional[List[str]] = None,
      predictions_to_stack: Optional[List[str]] = None,
      num_detections_key: Optional[str] = None
  ) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""

    metric_util.validate_object_detection_arguments(
        class_id=class_ids,
        class_weight=class_weights,
        area_range=area_range,
        max_num_detections=max_num_detections,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        output_name=output_name)

    if class_weights is None:
      class_weights = [1.0] * len(class_ids)

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=None,
        example_weighted=example_weighted,
        aggregation_type=None)

    computations = []
    recalls_keys = []
    for class_id, class_weight in zip(class_ids, class_weights):
      max_recall_name = metric_util.generate_private_name_from_arguments(
          AVERAGE_RECALL_NAME,
          iou_thresholds=iou_thresholds,
          class_id=class_id,
          class_weight=class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections)

      mr = COCOAverageRecall(
          iou_thresholds=iou_thresholds,
          class_id=class_id,
          class_weight=class_weight,
          area_range=area_range,
          max_num_detections=max_num_detections,
          name=max_recall_name,
          labels_to_stack=labels_to_stack,
          predictions_to_stack=predictions_to_stack,
          num_detections_key=num_detections_key)
      computations.extend(
          mr.computations(model_names=[model_name], output_names=[output_name]))
      recalls_keys.append(computations[-1].keys[-1])

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      recalls = [metrics[recalls_key] for recalls_key in recalls_keys]
      value = np.nanmean(recalls)
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations.append(derived_computation)
    return computations


metric_types.register_metric(COCOMeanAverageRecall)
