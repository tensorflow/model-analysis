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

from tensorflow_model_analysis.metrics import bleu, preprocessors, rouge
from tensorflow_model_analysis.metrics.attributions import (
    AttributionsMetric,
    MeanAbsoluteAttributions,
    MeanAttributions,
    TotalAbsoluteAttributions,
    TotalAttributions,
    has_attributions_metrics,
)
from tensorflow_model_analysis.metrics.calibration import (
    Calibration,
    MeanLabel,
    MeanPrediction,
)
from tensorflow_model_analysis.metrics.calibration_plot import CalibrationPlot
from tensorflow_model_analysis.metrics.confusion_matrix_metrics import (
    AUC,
    FN,
    FNR,
    FP,
    FPR,
    NPV,
    PPV,
    TN,
    TNR,
    TP,
    TPR,
    AUCCurve,
    AUCPrecisionRecall,
    AUCSummationMethod,
    BalancedAccuracy,
    BinaryAccuracy,
    ConfusionMatrixAtThresholds,
    DiagnosticOddsRatio,
    F1Score,
    FallOut,
    FalseDiscoveryRate,
    FalseNegatives,
    FalseOmissionRate,
    FalsePositives,
    FowlkesMallowsIndex,
    Informedness,
    Markedness,
    MatthewsCorrelationCoefficient,
    MaxRecall,
    MissRate,
    NegativeLikelihoodRatio,
    NegativePredictiveValue,
    PositiveLikelihoodRatio,
    Precision,
    PrecisionAtRecall,
    Prevalence,
    PrevalenceThreshold,
    Recall,
    RecallAtPrecision,
    SensitivityAtSpecificity,
    Specificity,
    SpecificityAtSensitivity,
    ThreatScore,
    TrueNegatives,
    TruePositives,
)
from tensorflow_model_analysis.metrics.confusion_matrix_plot import ConfusionMatrixPlot
from tensorflow_model_analysis.metrics.cross_entropy_metrics import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
)
from tensorflow_model_analysis.metrics.exact_match import ExactMatch
from tensorflow_model_analysis.metrics.example_count import ExampleCount
from tensorflow_model_analysis.metrics.flip_metrics import (
    BooleanFlipRates,
    NegToNegFlipRate,
    NegToPosFlipRate,
    PosToNegFlipRate,
    PosToPosFlipRate,
    SymmetricFlipRate,
)
from tensorflow_model_analysis.metrics.mean_regression_error import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogarithmicError,
)
from tensorflow_model_analysis.metrics.metric_specs import (
    default_binary_classification_specs,
    default_multi_class_classification_specs,
    default_regression_specs,
    metric_thresholds_from_metrics_specs,
    specs_from_metrics,
)
from tensorflow_model_analysis.metrics.metric_types import (
    CombinedFeaturePreprocessor,
    DerivedMetricComputation,
    FeaturePreprocessor,
    Metric,
    MetricComputation,
    MetricComputations,
    MetricKey,
    MetricsDict,
    PlotKey,
    Preprocessor,
    StandardMetricInputs,
    SubKey,
)
from tensorflow_model_analysis.metrics.metric_util import (
    merge_per_key_computations,
    to_label_prediction_example_weight,
    to_standard_metric_inputs,
)
from tensorflow_model_analysis.metrics.min_label_position import MinLabelPosition
from tensorflow_model_analysis.metrics.model_cosine_similarity import (
    ModelCosineSimilarity,
)
from tensorflow_model_analysis.metrics.multi_class_confusion_matrix_metrics import (
    NO_PREDICTED_CLASS_ID,
    MultiClassConfusionMatrixAtThresholds,
)
from tensorflow_model_analysis.metrics.multi_class_confusion_matrix_plot import (
    MultiClassConfusionMatrixPlot,
)
from tensorflow_model_analysis.metrics.multi_label_confusion_matrix_plot import (
    MultiLabelConfusionMatrixPlot,
)
from tensorflow_model_analysis.metrics.ndcg import NDCG
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_metrics import (
    ObjectDetectionMaxRecall,
    ObjectDetectionPrecision,
    ObjectDetectionPrecisionAtRecall,
    ObjectDetectionRecall,
    ObjectDetectionThresholdAtRecall,
)
from tensorflow_model_analysis.metrics.object_detection_confusion_matrix_plot import (
    ObjectDetectionConfusionMatrixPlot,
)
from tensorflow_model_analysis.metrics.object_detection_metrics import (
    COCOAveragePrecision,
    COCOAverageRecall,
    COCOMeanAveragePrecision,
    COCOMeanAverageRecall,
)
from tensorflow_model_analysis.metrics.prediction_difference_metrics import (
    SymmetricPredictionDifference,
)
from tensorflow_model_analysis.metrics.query_statistics import QueryStatistics
from tensorflow_model_analysis.metrics.score_distribution_plot import (
    ScoreDistributionPlot,
)
from tensorflow_model_analysis.metrics.semantic_segmentation_confusion_matrix_metrics import (
    SemanticSegmentationConfusionMatrix,
    SemanticSegmentationFalsePositive,
    SemanticSegmentationTruePositive,
)
from tensorflow_model_analysis.metrics.set_match_confusion_matrix_metrics import (
    SetMatchPrecision,
    SetMatchRecall,
)
from tensorflow_model_analysis.metrics.squared_pearson_correlation import (
    SquaredPearsonCorrelation,
)
from tensorflow_model_analysis.metrics.stats import Mean
from tensorflow_model_analysis.metrics.tjur_discrimination import (
    CoefficientOfDiscrimination,
    RelativeCoefficientOfDiscrimination,
)
from tensorflow_model_analysis.metrics.weighted_example_count import (
    WeightedExampleCount,
)

# TODO(b/143180976): Remove WeightedExampleCount.

__all__ = [
    "AttributionsMetric",
    "AUC",
    "AUCCurve",
    "AUCPrecisionRecall",
    "AUCSummationMethod",
    "BalancedAccuracy",
    "BinaryAccuracy",
    "BinaryCrossEntropy",
    "BooleanFlipRates",
    "Calibration",
    "CalibrationPlot",
    "CategoricalCrossEntropy",
    "COCOAveragePrecision",
    "COCOAverageRecall",
    "COCOMeanAveragePrecision",
    "COCOMeanAverageRecall",
    "CoefficientOfDiscrimination",
    "CombinedFeaturePreprocessor",
    "ConfusionMatrixAtThresholds",
    "ConfusionMatrixPlot",
    "default_binary_classification_specs",
    "default_multi_class_classification_specs",
    "default_regression_specs",
    "DerivedMetricComputation",
    "DiagnosticOddsRatio",
    "ExactMatch",
    "ExampleCount",
    "F1Score",
    "FallOut",
    "FalseDiscoveryRate",
    "FalseNegatives",
    "FalseOmissionRate",
    "FalsePositives",
    "FeaturePreprocessor",
    "FN",
    "FNR",
    "FowlkesMallowsIndex",
    "FP",
    "FPR",
    "has_attributions_metrics",
    "Informedness",
    "Markedness",
    "MatthewsCorrelationCoefficient",
    "MaxRecall",
    "Mean",
    "MeanAbsoluteAttributions",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanAttributions",
    "MeanLabel",
    "MeanPrediction",
    "MeanSquaredError",
    "MeanSquaredLogarithmicError",
    "merge_per_key_computations",
    "Metric",
    "metric_thresholds_from_metrics_specs",
    "MetricComputation",
    "MetricComputations",
    "MetricKey",
    "MetricsDict",
    "MinLabelPosition",
    "MissRate",
    "MultiClassConfusionMatrixAtThresholds",
    "MultiClassConfusionMatrixPlot",
    "MultiLabelConfusionMatrixPlot",
    "NDCG",
    "NegativeLikelihoodRatio",
    "NegativePredictiveValue",
    "NO_PREDICTED_CLASS_ID",
    "NPV",
    "ObjectDetectionConfusionMatrixPlot",
    "ObjectDetectionMaxRecall",
    "ObjectDetectionPrecision",
    "ObjectDetectionPrecisionAtRecall",
    "ObjectDetectionRecall",
    "ObjectDetectionThresholdAtRecall",
    "PlotKey",
    "PositiveLikelihoodRatio",
    "PPV",
    "Precision",
    "PrecisionAtRecall",
    "Preprocessor",
    "Prevalence",
    "PrevalenceThreshold",
    "QueryStatistics",
    "Recall",
    "RecallAtPrecision",
    "RelativeCoefficientOfDiscrimination",
    "ScoreDistributionPlot",
    "SemanticSegmentationConfusionMatrix",
    "SemanticSegmentationFalsePositive",
    "SemanticSegmentationTruePositive",
    "SensitivityAtSpecificity",
    "SetMatchPrecision",
    "SetMatchRecall",
    "Specificity",
    "SpecificityAtSensitivity",
    "specs_from_metrics",
    "SquaredPearsonCorrelation",
    "StandardMetricInputs",
    "SubKey",
    "SymmetricPredictionDifference",
    "ThreatScore",
    "TN",
    "TNR",
    "to_label_prediction_example_weight",
    "to_standard_metric_inputs",
    "TotalAbsoluteAttributions",
    "TotalAttributions",
    "TP",
    "TPR",
    "TrueNegatives",
    "TruePositives",
    "WeightedExampleCount",
]
