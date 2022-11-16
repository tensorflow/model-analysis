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
"""Object Detection Confusion matrix Plot."""

from typing import List, Optional, Tuple

from tensorflow_model_analysis.metrics import confusion_matrix_plot
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import preprocessors
from tensorflow_model_analysis.proto import config_pb2

DEFAULT_NUM_THRESHOLDS = 1000

CONFUSION_MATRIX_PLOT_NAME = 'confusion_matrix_plot'


class ObjectDetectionConfusionMatrixPlot(
    confusion_matrix_plot.ConfusionMatrixPlot):
  """Object Detection Confusion matrix plot."""

  def __init__(self,
               num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
               name: str = CONFUSION_MATRIX_PLOT_NAME,
               iou_threshold: Optional[float] = None,
               class_id: Optional[int] = None,
               class_weight: Optional[float] = None,
               area_range: Optional[Tuple[float, float]] = None,
               max_num_detections: Optional[int] = None,
               labels_to_stack: Optional[List[str]] = None,
               predictions_to_stack: Optional[List[str]] = None,
               num_detections_key: Optional[str] = None):
    """Initializes confusion matrix plot for object detection.

    The metric supports using multiple outputs to form the labels/predictions if
    the user specifies the label/predcition keys to stack. In this case, the
    metric is not expected to work with multi-outputs. The metric only supports
    multi outputs if the output of model is already pre-stacked in the expected
    format, i.e. ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'] for labels and
    ['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence scores'] for
    predictions.

    Args:
      num_thresholds: Number of thresholds to use when discretizing the curve.
        Values must be > 1. Defaults to 1000.
      name: Metric name.
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
        num_thresholds=num_thresholds,
        name=name,
        class_id=class_id,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_num_detections=max_num_detections,
        class_weight=class_weight,
        labels_to_stack=labels_to_stack,
        predictions_to_stack=predictions_to_stack,
        num_detections_key=num_detections_key)

  def _confusion_matrix_plot(self,
                             num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
                             name: str = CONFUSION_MATRIX_PLOT_NAME,
                             eval_config: Optional[
                                 config_pb2.EvalConfig] = None,
                             model_name: str = '',
                             output_name: str = '',
                             iou_threshold: Optional[float] = None,
                             class_id: Optional[int] = None,
                             class_weight: Optional[float] = None,
                             area_range: Optional[Tuple[float, float]] = None,
                             max_num_detections: Optional[int] = None,
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

    return super()._confusion_matrix_plot(
        num_thresholds=num_thresholds,
        name=name,
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        preprocessors=[preprocessor],
        **kwargs)


metric_types.register_metric(ObjectDetectionConfusionMatrixPlot)
