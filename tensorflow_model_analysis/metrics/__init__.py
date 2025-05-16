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
"""Init module for TensorFlow Model Analysis metrics."""

from tensorflow_model_analysis.metrics import bleu
from tensorflow_model_analysis.metrics import preprocessors
from tensorflow_model_analysis.metrics import rouge
from tensorflow_model_analysis.metrics.attributions import AttributionsMetric
from tensorflow_model_analysis.metrics.attributions import has_attributions_metrics
from tensorflow_model_analysis.metrics.attributions import MeanAbsoluteAttributions
from tensorflow_model_analysis.metrics.attributions import MeanAttributions
from tensorflow_model_analysis.metrics.attributions import TotalAbsoluteAttributions
from tensorflow_model_analysis.metrics.attributions import TotalAttributions
from tensorflow_model_analysis.metrics.calibration import Calibration
from tensorflow_model_analysis.metrics.calibration import MeanLabel
from tensorflow_model_analysis.metrics.calibration import MeanPrediction
from tensorflow_model_analysis.metrics.calibration_plot import CalibrationPlot
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import AUC
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import AUCCurve
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import AUCPrecisionRecall
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import AUCSummationMethod
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import BalancedAccuracy
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import BinaryAccuracy
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import ConfusionMatrixAtThresholds
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import DiagnosticOddsRatio
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import F1Score
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FallOut
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FalseDiscoveryRate
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FalseNegatives
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FalseOmissionRate
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FalsePositives
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FN
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FNR
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FowlkesMallowsIndex
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FP
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import FPR
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import Informedness
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import Markedness
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import MatthewsCorrelationCoefficient
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import MaxRecall
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import MissRate
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import NegativeLikelihoodRatio
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import NegativePredictiveValue
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import NPV
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import PositiveLikelihoodRatio
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import PPV
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import Precision
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import PrecisionAtRecall
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import Prevalence
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import PrevalenceThreshold
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import Recall
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import RecallAtPrecision
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import SensitivityAtSpecificity
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import Specificity
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import SpecificityAtSensitivity
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import ThreatScore
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import TN
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import TNR
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import TP
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import TPR
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import TrueNegatives
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import TruePositives
from tensorflow_model_analysis.metrics.confusion_matrix_plot import ConfusionMatrixPlot
from tensorflow_model_analysis.metrics.cross_entropy_metrics import BinaryCrossEntropy
from tensorflow_model_analysis.metrics.cross_entropy_metrics import CategoricalCrossEntropy
from tensorflow_model_analysis.metrics.exact_match import ExactMatch
from tensorflow_model_analysis.metrics.example_count import ExampleCount
from tensorflow_model_analysis.metrics.flip_metrics import BooleanFlipRates
from tensorflow_model_analysis.metrics.flip_metrics import NegToNegFlipRate
from tensorflow_model_analysis.metrics.flip_metrics import NegToPosFlipRate
from tensorflow_model_analysis.metrics.flip_metrics import PosToNegFlipRate
from tensorflow_model_analysis.metrics.flip_metrics import PosToPosFlipRate
from tensorflow_model_analysis.metrics.flip_metrics import SymmetricFlipRate
from tensorflow_model_analysis.metrics.mean_regression_error import MeanAbsoluteError
from tensorflow_model_analysis.metrics.mean_regression_error import MeanAbsolutePercentageError
from tensorflow_model_analysis.metrics.mean_regression_error import MeanSquaredError
from tensorflow_model_analysis.metrics.mean_regression_error import MeanSquaredLogarithmicError
from tensorflow_model_analysis.metrics.metric_specs import default_binary_classification_specs
from tensorflow_model_analysis.metrics.metric_specs import default_multi_class_classification_specs
from tensorflow_model_analysis.metrics.metric_specs import default_regression_specs
from tensorflow_model_analysis.metrics.metric_specs import metric_thresholds_from_metrics_specs
from tensorflow_model_analysis.metrics.metric_specs import specs_from_metrics
from tensorflow_model_analysis.metrics.metric_types import CombinedFeaturePreprocessor
from tensorflow_model_analysis.metrics.metric_types import DerivedMetricComputation
from tensorflow_model_analysis.metrics.metric_types import FeaturePreprocessor
from tensorflow_model_analysis.metrics.metric_types import Metric
from tensorflow_model_analysis.metrics.metric_types import MetricComputation
from tensorflow_model_analysis.metrics.metric_types import MetricComputations
from tensorflow_model_analysis.metrics.metric_types import MetricKey
from tensorflow_model_analysis.metrics.metric_types import MetricsDict
from tensorflow_model_analysis.metrics.metric_types import PlotKey
from tensorflow_model_analysis.metrics.metric_types import Preprocessor
from tensorflow_model_analysis.metrics.metric_types import StandardMetricInputs
from tensorflow_model_analysis.metrics.metric_types import SubKey
from tensorflow_model_analysis.metrics.metric_util import merge_per_key_computations
from tensorflow_model_analysis.metrics.metric_util import to_label_prediction_example_weight
from tensorflow_model_analysis.metrics.metric_util import to_standard_metric_inputs
from tensorflow_model_analysis.metrics.min_label_position import MinLabelPosition
from tensorflow_model_analysis.metrics.model_cosine_similarity import ModelCosineSimilarity
from tensorflow_model_analysis.metrics.multi_class_confusion_matrix_metrics import MultiClassConfusionMatrixAtThresholds
from tensorflow_model_analysis.metrics.multi_class_confusion_matrix_metrics import NO_PREDICTED_CLASS_ID
from tensorflow_model_analysis.metrics.multi_class_confusion_matrix_plot import MultiClassConfusionMatrixPlot
from tensorflow_model_analysis.metrics.multi_label_confusion_matrix_plot import MultiLabelConfusionMatrixPlot
from tensorflow_model_analysis.metrics.ndcg import NDCG
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_metrics import ObjectDetectionMaxRecall
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_metrics import ObjectDetectionPrecision
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_metrics import ObjectDetectionPrecisionAtRecall
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_metrics import ObjectDetectionRecall
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_metrics import ObjectDetectionThresholdAtRecall
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_plot import ObjectDetectionConfusionMatrixPlot
from tensorflow_model_analysis.metrics.object_detection_metrics import COCOAveragePrecision
from tensorflow_model_analysis.metrics.object_detection_metrics import COCOAverageRecall
from tensorflow_model_analysis.metrics.object_detection_metrics import COCOMeanAveragePrecision
from tensorflow_model_analysis.metrics.object_detection_metrics import COCOMeanAverageRecall
from tensorflow_model_analysis.metrics.prediction_difference_metrics import SymmetricPredictionDifference
from tensorflow_model_analysis.metrics.query_statistics import QueryStatistics
from tensorflow_model_analysis.metrics.score_distribution_plot import ScoreDistributionPlot
from tensorflow_model_analysis.metrics.semantic_segmentation_confusion_matrix_metrics import SemanticSegmentationConfusionMatrix
from tensorflow_model_analysis.metrics.semantic_segmentation_confusion_matrix_metrics import SemanticSegmentationFalsePositive
from tensorflow_model_analysis.metrics.semantic_segmentation_confusion_matrix_metrics import SemanticSegmentationTruePositive
from tensorflow_model_analysis.metrics.set_match_confusion_matrix_metrics import SetMatchPrecision
from tensorflow_model_analysis.metrics.set_match_confusion_matrix_metrics import SetMatchRecall
from tensorflow_model_analysis.metrics.squared_pearson_correlation import SquaredPearsonCorrelation
from tensorflow_model_analysis.metrics.stats import Mean
from tensorflow_model_analysis.metrics.tjur_discrimination import CoefficientOfDiscrimination
from tensorflow_model_analysis.metrics.tjur_discrimination import RelativeCoefficientOfDiscrimination
from tensorflow_model_analysis.metrics.weighted_example_count import WeightedExampleCount

# TODO(b/143180976): Remove WeightedExampleCount.

__all__ = [
  'AttributionsMetric',
  'AUC',
  'AUCCurve',
  'AUCPrecisionRecall',
  'AUCSummationMethod',
  'BalancedAccuracy',
  'BinaryAccuracy',
  'BinaryCrossEntropy',
  'BooleanFlipRates',
  'Calibration',
  'CalibrationPlot',
  'CategoricalCrossEntropy',
  'COCOAveragePrecision',
  'COCOAverageRecall',
  'COCOMeanAveragePrecision',
  'COCOMeanAverageRecall',
  'CoefficientOfDiscrimination',
  'CombinedFeaturePreprocessor',
  'ConfusionMatrixAtThresholds',
  'ConfusionMatrixPlot',
  'default_binary_classification_specs',
  'default_multi_class_classification_specs',
  'default_regression_specs',
  'DerivedMetricComputation',
  'DiagnosticOddsRatio',
  'ExactMatch',
  'ExampleCount',
  'F1Score',
  'FallOut',
  'FalseDiscoveryRate',
  'FalseNegatives',
  'FalseOmissionRate',
  'FalsePositives',
  'FeaturePreprocessor',
  'FN',
  'FNR',
  'FowlkesMallowsIndex',
  'FP',
  'FPR',
  'has_attributions_metrics',
  'Informedness',
  'Markedness',
  'MatthewsCorrelationCoefficient',
  'MaxRecall',
  'Mean',
  'MeanAbsoluteAttributions',
  'MeanAbsoluteError',
  'MeanAbsolutePercentageError',
  'MeanAttributions',
  'MeanLabel',
  'MeanPrediction',
  'MeanSquaredError',
  'MeanSquaredLogarithmicError',
  'merge_per_key_computations',
  'Metric',
  'metric_thresholds_from_metrics_specs',
  'MetricComputation',
  'MetricComputations',
  'MetricKey',
  'MetricsDict',
  'MinLabelPosition',
  'MissRate',
  'MultiClassConfusionMatrixAtThresholds',
  'MultiClassConfusionMatrixPlot',
  'MultiLabelConfusionMatrixPlot',
  'NDCG',
  'NegativeLikelihoodRatio',
  'NegativePredictiveValue',
  'NO_PREDICTED_CLASS_ID',
  'NPV',
  'ObjectDetectionConfusionMatrixPlot',
  'ObjectDetectionMaxRecall',
  'ObjectDetectionPrecision',
  'ObjectDetectionPrecisionAtRecall',
  'ObjectDetectionRecall',
  'ObjectDetectionThresholdAtRecall',
  'PlotKey',
  'PositiveLikelihoodRatio',
  'PPV',
  'Precision',
  'PrecisionAtRecall',
  'Preprocessor',
  'Prevalence',
  'PrevalenceThreshold',
  'QueryStatistics',
  'Recall',
  'RecallAtPrecision',
  'RelativeCoefficientOfDiscrimination',
  'ScoreDistributionPlot',
  'SemanticSegmentationConfusionMatrix',
  'SemanticSegmentationFalsePositive',
  'SemanticSegmentationTruePositive',
  'SensitivityAtSpecificity',
  'SetMatchPrecision',
  'SetMatchRecall',
  'Specificity',
  'SpecificityAtSensitivity',
  'specs_from_metrics',
  'SquaredPearsonCorrelation',
  'StandardMetricInputs',
  'SubKey',
  'SymmetricPredictionDifference',
  'ThreatScore',
  'TN',
  'TNR',
  'to_label_prediction_example_weight',
  'to_standard_metric_inputs',
  'TotalAbsoluteAttributions',
  'TotalAttributions',
  'TP',
  'TPR',
  'TrueNegatives',
  'TruePositives',
  'WeightedExampleCount'
]
