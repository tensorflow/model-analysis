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
"""Build confusion matrices for semantic segmentation."""

import abc
from typing import Any, Dict, Iterable, List, Optional, Union
import apache_beam as beam
import numpy as np
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import preprocessors
from tensorflow_model_analysis.proto import config_pb2

# default values for object detection settings
_DEFAULT_IOU_THRESHOLD = 0.5
_DEFAULT_AREA_RANGE = (0, float('inf'))

SEMANTIC_SEGMENTATION_TRUE_POSITIVES_NAME = (
    'semantic_segmentation_true_positives'
)
SEMANTIC_SEGMENTATION_FALSE_POSITIVES_NAME = (
    'semantic_segmentation_false_positives'
)
# The confusion matrix metric is supposed to be a private metric. It should only
# be as the intermediate results for other metrics. To access the entries,
# please use metrics like TruePositive, TrueNegative, etc.
SEMANTIC_SEGMENTATION_CONFUSION_MATRIX_NAME = (
    '_semantic_segmentation_confusion_matrix'
)


class SemanticSegmentationConfusionMatrix(metric_types.Metric):
  """Computes confusion matrices for semantic segmentation."""

  def __init__(
      self,
      class_ids: List[int],
      ground_truth_key: str,
      prediction_key: str,
      decode_ground_truth: bool = True,
      decode_prediction: bool = False,
      ignore_ground_truth_id: Optional[int] = None,
      name: Optional[str] = None,
  ):
    """Initializes PrecisionAtRecall metric.

    Args:
      class_ids: the class ids for calculating metrics.
      ground_truth_key: the key for storing the ground truth of encoded image
        with class ids.
      prediction_key: the key for storing the predictions of encoded image with
        class ids.
      decode_ground_truth: If true, the ground truth is assumed to be bytes of
        images and will be decoded. By default it is true assuming the label is
        the bytes of image.
      decode_prediction: If true, the prediction is assumed to be bytes of
        images and will be decoded. By default it is false assuming the model
        outputs numpy arrays or tensors.
      ignore_ground_truth_id: (Optional) The id of ground truth to be ignored.
      name: (Optional) string name of the metric instance.
    """

    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        name=name,
        class_ids=class_ids,
        ground_truth_key=ground_truth_key,
        prediction_key=prediction_key,
        decode_ground_truth=decode_ground_truth,
        decode_prediction=decode_prediction,
        ignore_ground_truth_id=ignore_ground_truth_id,
    )

  def _default_name(self) -> str:
    return SEMANTIC_SEGMENTATION_CONFUSION_MATRIX_NAME

  def _metric_computations(
      self,
      class_ids: List[int],
      ground_truth_key: str,
      prediction_key: str,
      decode_ground_truth: bool = True,
      decode_prediction: bool = False,
      ignore_ground_truth_id: Optional[int] = None,
      name: Optional[str] = None,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False,
      **kwargs,
  ) -> metric_types.MetricComputations:
    preprocessor = preprocessors.DecodeImagePreprocessor(
        ground_truth_key=ground_truth_key,
        prediction_key=prediction_key,
        decode_ground_truth=decode_ground_truth,
        decode_prediction=decode_prediction,
        name=name,
        model_name=model_name,
    )
    # The model output is the key to store images and other informations.
    # The output name should not be set.
    # sub_key should not be set. The class_id info will be encoded in sub_key
    # in the combiners.
    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name='',
        sub_key=None,
        example_weighted=example_weighted,
    )

    return [
        metric_types.MetricComputation(
            keys=[key],
            preprocessors=[preprocessor],
            combiner=_SemanticSegmentationConfusionMatrixCombiner(
                key=key,
                class_ids=class_ids,
                ignore_ground_truth_id=ignore_ground_truth_id,
            ),
        )
    ]


class _SemanticSegmentationConfusionMatrixCombiner(beam.CombineFn):
  """Combines semantic segmentation confusion matrices."""

  def __init__(
      self,
      key: metric_types.MetricKey,
      class_ids: List[int],
      ignore_ground_truth_id: Optional[int] = None,
  ):
    """Initializes the semantic segmentation confusion matrix combiner.

    Args:
     key: The metric key to identify the metric output.
     class_ids: The ids of classes to calculate metrics.
     ignore_ground_truth_id: The id of the ignored class. It could be used for
       the class that should not be counted (e.g. masked pixels).
    """
    self._key = key
    self._class_ids = class_ids
    self._ignore_ground_truth_id = ignore_ground_truth_id

  def create_accumulator(self) -> Dict[int, binary_confusion_matrices.Matrix]:
    return {}

  def add_input(
      self,
      accumulator: Dict[int, binary_confusion_matrices.Matrix],
      element: metric_types.StandardMetricInputs,
  ) -> Dict[int, binary_confusion_matrices.Matrix]:
    ground_truth = element.get_labels()
    prediction = element.get_predictions()

    if ground_truth.ndim != 2:
      raise ValueError(
          'The ground truth should be in 2d. '
          f'But the shape is {ground_truth.shape}'
      )
    if ground_truth.shape != prediction.shape:
      raise ValueError(
          f'The shape of prediction {prediction.shape} does not'
          f' match the shape of ground truth {ground_truth.shape}'
      )

    for class_id in self._class_ids:
      class_true_positive = np.sum(
          np.logical_and(ground_truth == class_id, prediction == class_id)
      )
      class_false_positive = np.sum(
          np.logical_and(
              np.logical_and(ground_truth != class_id, prediction == class_id),
              ground_truth != self._ignore_ground_truth_id,
          )
      )
      class_false_negative = np.sum(
          np.logical_and(ground_truth == class_id, prediction != class_id)
      )
      class_true_negative = np.sum(
          np.logical_and(
              np.logical_and(ground_truth != class_id, prediction != class_id),
              ground_truth != self._ignore_ground_truth_id,
          )
      )

      class_confusion_matrix = binary_confusion_matrices.Matrix(
          tp=class_true_positive,
          tn=class_true_negative,
          fp=class_false_positive,
          fn=class_false_negative,
      )
      if class_id not in accumulator:
        accumulator[class_id] = class_confusion_matrix
      else:
        accumulator[class_id] = binary_confusion_matrices.Matrix(
            tp=accumulator[class_id].tp + class_confusion_matrix.tp,
            tn=accumulator[class_id].tn + class_confusion_matrix.tn,
            fp=accumulator[class_id].fp + class_confusion_matrix.fp,
            fn=accumulator[class_id].fn + class_confusion_matrix.fn,
        )
    return accumulator

  def merge_accumulators(
      self,
      accumulators: Iterable[Dict[int, binary_confusion_matrices.Matrix]],
  ) -> Dict[int, binary_confusion_matrices.Matrix]:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      for class_id, confusion_matrix in accumulator.items():
        if class_id in result:
          result[class_id] = binary_confusion_matrices.Matrix(
              tp=result[class_id].tp + confusion_matrix.tp,
              tn=result[class_id].tn + confusion_matrix.tn,
              fp=result[class_id].fp + confusion_matrix.fp,
              fn=result[class_id].fn + confusion_matrix.fn,
          )
        else:
          result[class_id] = confusion_matrix
    return result

  def extract_output(
      self, accumulator: Dict[int, binary_confusion_matrices.Matrix]
  ) -> Dict[metric_types.MetricKey, binary_confusion_matrices.Matrix]:
    result = {}
    for class_id, confusion_matrix_matrix in accumulator.items():
      new_key = self._key._replace(
          sub_key=metric_types.SubKey(class_id=class_id)
      )
      # In semantic segmentation metrics, there is no confidence score and thus
      # no thresholds. It is set to 0 as default to reuse the binary confusion
      # matrices.
      matrices = binary_confusion_matrices.Matrix(
          tp=confusion_matrix_matrix.tp,
          tn=confusion_matrix_matrix.tn,
          fp=confusion_matrix_matrix.fp,
          fn=confusion_matrix_matrix.fn,
      )
      result[new_key] = matrices
    return result


metric_types.register_metric(SemanticSegmentationConfusionMatrix)


class SemanticSegmentationConfusionMatrixMetricBase(
    metric_types.Metric, metaclass=abc.ABCMeta
):
  """The base metric for semantic segmentation confusion matrix based metrics.

  This is the base metric for other metrics such as true postive, true negative,
  false positvie and false negative.
  """

  def __init__(
      self,
      class_ids: List[int],
      ground_truth_key: str,
      prediction_key: str,
      decode_ground_truth: bool = True,
      decode_prediction: bool = False,
      ignore_ground_truth_id: Optional[int] = None,
      name: Optional[str] = None,
  ):
    """Initializes PrecisionAtRecall metric.

    Args:
      class_ids: the class ids for calculating metrics.
      ground_truth_key: the key for storing the ground truth of encoded image
        with class ids.
      prediction_key: the key for storing the predictions of encoded image with
        class ids.
      decode_ground_truth: If true, the ground truth is assumed to be bytes of
        images and will be decoded. By default it is true assuming the label is
        the bytes of image.
      decode_prediction: If true, the prediction is assumed to be bytes of
        images and will be decoded. By default it is false assuming the model
        outputs numpy arrays or tensors.
      ignore_ground_truth_id: (Optional) The id of ground truth to be ignored.
      name: (Optional) string name of the metric instance.
    """

    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        name=name,
        class_ids=class_ids,
        ground_truth_key=ground_truth_key,
        prediction_key=prediction_key,
        decode_ground_truth=decode_ground_truth,
        decode_prediction=decode_prediction,
        ignore_ground_truth_id=ignore_ground_truth_id,
    )

  @abc.abstractmethod
  def _default_name(self) -> str:
    """Returns the default metric name."""
    raise NotImplementedError('Must have a default name for the metric.')

  @abc.abstractmethod
  def _metric_value(
      self,
      matrix: binary_confusion_matrices.Matrix,
  ) -> Union[float, np.ndarray]:
    """Returns the metric value based the confusion matrix.

       Subclasses must override this method.
    Args:
      matrix: The matrix to calculate derived values.

    Return: The values calculated based on a confusion matrix.
    """
    raise NotImplementedError('Must be implemented to return a metric value')

  def _metric_computations(
      self,
      class_ids: List[int],
      ground_truth_key: str,
      prediction_key: str,
      decode_ground_truth: bool = True,
      decode_prediction: bool = False,
      ignore_ground_truth_id: Optional[int] = None,
      name: Optional[str] = None,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      output_name: str = '',
      example_weighted: bool = False,
      **kwargs,
  ) -> metric_types.MetricComputations:
    # generates private name to distinguish semantic segmentation confusion
    # matrix from different configs.
    semantic_segmentation_confusion_matrix_name = (
        metric_util.generate_private_name_from_arguments(
            class_ids=class_ids,
            ground_truth_key=ground_truth_key,
            prediction_key=prediction_key,
            decode_ground_truth=decode_ground_truth,
            decode_prediction=decode_prediction,
            ignore_ground_truth_id=ignore_ground_truth_id,
            name=name,
        )
    )
    maxtrix_computation = SemanticSegmentationConfusionMatrix(
        class_ids=class_ids,
        ground_truth_key=ground_truth_key,
        prediction_key=prediction_key,
        decode_ground_truth=decode_ground_truth,
        decode_prediction=decode_prediction,
        ignore_ground_truth_id=ignore_ground_truth_id,
        name=semantic_segmentation_confusion_matrix_name,
    )
    computations = maxtrix_computation.computations(
        model_names=[model_name],
        output_names=[output_name],
        example_weighted=example_weighted,
    )
    # This is the key to metric output for the entire matrix computation.
    # sub_key/class_id info is not encoded in the metric key yet.
    matrix_key = computations[-1].keys[-1]

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
    )

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      derived_output = {}
      for class_id in class_ids:
        class_matrix_key = matrix_key._replace(
            sub_key=metric_types.SubKey(class_id=class_id)
        )
        class_output_key = key._replace(
            sub_key=metric_types.SubKey(class_id=class_id)
        )
        derived_output[class_output_key] = self._metric_value(
            metrics[class_matrix_key]
        )
      return derived_output

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result
    )
    computations.append(derived_computation)

    return computations


class SemanticSegmentationTruePositive(
    SemanticSegmentationConfusionMatrixMetricBase
):
  """Calculates the true postive for semantic segmentation."""

  def _default_name(self) -> str:
    return SEMANTIC_SEGMENTATION_TRUE_POSITIVES_NAME

  def _metric_value(self, matrix: binary_confusion_matrices.Matrix) -> float:
    return matrix.tp


metric_types.register_metric(SemanticSegmentationTruePositive)


class SemanticSegmentationFalsePositive(
    SemanticSegmentationConfusionMatrixMetricBase
):
  """Calculates the true postive for semantic segmentation."""

  def _default_name(self) -> str:
    return SEMANTIC_SEGMENTATION_FALSE_POSITIVES_NAME

  def _metric_value(self, matrix: binary_confusion_matrices.Matrix) -> float:
    return matrix.fp


metric_types.register_metric(SemanticSegmentationFalsePositive)
