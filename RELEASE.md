<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug fixes and other Changes

## Breaking Changes

## Deprecations

# Version 0.48.0

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Depends on `tensorflow>=2.17,<2.18`.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on
    `protobuf>=4.21.6,<6.0.0` for 3.9 and 3.10.
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on
    `apache-beam[gcp]>=2.50.0,<2.51` for 3.9 and 3.10.
*   macOS wheel publishing is temporarily paused due to missing ARM64 support.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.47.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Removing addons from __init__.py as it's deprecated with Eval Saved Model.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.47.0

## Major Features and Improvements

*   Adds `False{Negative,Positive}FeatureSampler` metrics.
*   Adds `RecallAtFalsePositiveRate` metrics.
*   Adds 'NegToNegFlipRate', 'NegToPosFlipRate', 'PosToNegFlipRate',
    'PosToPosFlipRate', and 'SymmetricFlipRate' metrics.
*   Graduates dataframe module to core library as tfma.dataframe.

## Bug fixes and other Changes

*   Adds support for absolute change threshold validation on native diff metrics
    like flip rate metrics (e.g. `SymmetricFlipRate`) and
    `ModelCosineSimilarity`.
*   Fixes a bug by ensuring feature values are always numpy arrays.
*   Modifies a ROUGE Test to be compatible with NumPy v2.0.1.
*   Remove keras_util_test.py which is based on estimator models.
*   Remove dependency on eval_saved_model encodings.
*   Downloads `punkt_tab` in Rouge metric.
*   Depends on `tensorflow 2.16`
*   Relax dependency on Protobuf to include version 5.x

## Breaking Changes

*   Removing legacy_predict_extractor in model_eval_lib.py.
*   Removing post_export_metrics support in model_eval_lib.py.
*   Removing eval saved model related API in
    metrics_plots_and_validations_evaluator.py
*   Removing legacy_metrics_and_plots_evaluator in model_eval_lib.py.
*   Removes legacy_metrics_and_plots_evaluator from the public API of TFMA
    evaluator.
*   Removing eval_saved_model related API in model_util.py, estimator related
    functions are no longer supported.
*   Removing legacy_metrics_and_plots_evaluator in TFMA OSS.
*   Removing legacy_aggregate in TFMA.
*   Remove legacy_query_based_metrics_evaluator.py and its test.
*   Remove model_agnostic_eval, export, exporter, and post_export_metrics which are based on estimators.
*   Remove eval_saved_model_util.py and its test.
*   Remove contrib model_eval_lib and export and their tests.
*   Remove all eval_saved_model files.

## Deprecations

*   Migrate common utils in eval_saved_model testutils to utils/test_util.py.
    This enables further deprecation of eval saved model.
*   Deprecate legacy estimator related tests in predictions_extractor_test.py

# Version 0.46.0

## Major Features and Improvements

*   Removes the metrics modules from experimental now that it is migrated to
    [py-ml-metrics](https://pypi.org/project/py-ml-metrics/) package.
*   Adds Constituent Flip Rate Metrics: SymmetricFlipRate, NegToNegFlipRate,
    NegToPosFlipRate, PosToNegFlipRate, PosToPosFlipRate.
*   Adds Model Cosine Similarity Metrics.
*   Depend on tensorflow-estimator package explicitly.

## Bug fixes and other Changes

*   Fix the bug about batching unsized numpy arrays.

## Breaking Changes

*   Removes `attrs` requirement.
*   Consolidate Matrix definition for semantic segmentation confusion matrix
    metrics.
*   Provide AggregateFn and interface and default call impl to adapt TFMA
    metrics combiner for in-process call.
*   Move Mean metrics from experimental to metrics.
*   Fix the bug of size estimator failure.
*   Depends on `tensorflow>=2.15.0,<2.16`.
*   Fix the failure in testMeanAttributions.
*   Fix the input type mismatch in metric_specs_tests between bool and None.
*   Fix the failure in the slice test due to beam type hints check.
*   Fix the failure in metric_specs test, all TFMA deps on keras are
    keras 2.
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on
    `apache-beam[gcp]>=2.47.0,<3` for 3.9 and 3.10.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.
*   Update the minimum Bazel version required to build TFMA to 6.1.0
*   Refactors BooleanFlipRates computations to a combiner (flip_counts) and a
    DerivedMetricComputation (flip_rates).

## Deprecations

*   Deprecated python 3.8 support.

# Version 0.45.0

## Major Features and Improvements

*   Add F1, False positive rate, and Accuracy into the confusion matrix plot.
*   Add support for setting top_k and class_id at the same time for confusion
    matrix metrics.
*   Add the false positive for semantic segmentation metrics.
*   Add Mean Metric (experimental) which calculates the mean of any feature. *.
    Adds support of `output_keypath` to ModelSignatureDoFn to explicitly set a
    chain of output keys in the multi-level dict (extracts). Adds output_keypath
    to common prediction extractors.
*   Add Mean Metric (experimental) which calculates the mean of any feature. *.
    Adds support of `output_keypath` to ModelSignatureDoFn to explicitly set a
    chain of output keys in the multi-level dict (extracts). Adds output_keypath
    to common prediction extractors.
*   Add ROUGE Metrics.
*   Add BLEU Metric.
*   Refactor Binary Confusion Matrices to use Binary Confusion Matrices
    Computations.

## Bug fixes and other Changes

*   Fix the bug that SetMatchRecall is always 1 when top_k is set.
*   Depends on `pyarrow>=10,<11`.
*   Depends on `apache-beam>=2.47,<3`.
*   Depends on `numpy>=1.23.0`.
*   Depends on `tensorflow>=2.13.0,<3`.
*   Add 'tfma_eval' model_type in model_specs as the identifier for
    eval_saved_model, allowing signature='eval' to now be used with other model
    types.
*   Add "materialized_prediction" model type to allow users bypassing model
    inference explicitly.

## Breaking Changes

*   Depend on PIL for image related metrics.
*   Separate extract_key from signature names in `ModelSignaturesDoFn`.

## Deprecations

*   N/A

# Version 0.44.0

## Major Features and Improvements

*   Add BinaryCrossEntropy and CategoricalCrossEntropy.
*   Add MeanAbsolutePercentageError and MeanSquaredLogarithmicError
*   Add SetMatchPrecision and SetMatchRecall
*   Add SemanticSegmentationConfusionMatrix

## Bug fixes and other Changes

*   Fix for jupiter notebook
*   Fix element dimension inconsistency when some of the extracts have missing
    key.
*   Add public visibility to the servo beam extractor.
*   Fix for bug where binary_confusion_matrices with different class_weights are
    considered identical and deduplicated.
*   Fixes bug where positive / negative axes labels are reversed in prediction
    distribution plot.
*   Depends on `numpy~=1.22.0`.
*   Modify ExampleCount to not depend on labels and predictions.
*   Add class_id info into sub_key in metric_key for object detection confusion
    matrix metrics.
*   Add class_id info into sub_key in plot_key for object detection confusion
    matrix plot.
*   Fix a bug that auto_pivot dropped nan when deciding which columns are
    multivalent for pivoting.
*   Depends on `tensorflow>=2.12.0,<2.13`.
*   Depends on `protobuf>=3.20.3,<5`.
*   Depends on `tfx-bsl>=1.13.0,<1.14.0`.
*   Depends on `tensorflow-metadata>=1.13.1,<1.14.0`.
*   Add name for each plots.

## Breaking Changes

*   N/A

## Deprecations

Deprecated python3.7 support.

# Version 0.43.0

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Depends on `tensorflow>=2.11,<3`
*   Depends on `tfx-bsl>=1.2.0,<1.13.0`.
*   Depends on `tensorflow-metadata>=1.12.0,<1.13.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.42.0

## Major Features and Improvements

*   This is the last version that supports TensorFlow 1.15.x. TF 1.15.x support
    will be removed in the next version. Please check the
    [TF2 migration guide](https://www.tensorflow.org/guide/migrate) to migrate
    to TF2.
*   Add BooleanFlipRate metric for comparing thresholded predictions between
    multiple models.
*   Add CounterfactualPredictionsExtractor for computing predictions on modified
    inputs.
*   Add MeanAbsoluteError and MeanSquaredError

## Bug fixes and other Changes

*   Add support for parsing the Predict API prediction log output to the
    experimental TFX-BSL PredictionsExtractor implementation.
*   Add support for parsing the Classification API prediction log output to the
    experimental TFX-BSL PredictionsExtractor implementation.
*   Update remaining predictions_extractor_test.py tests to cover
    PredictionsExtractorOSS. Fixes a pytype bug related to multi tensor output.
*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.10,<3`

*   Apply changes in the latest Chrome browser

*   Add InferneceInterface to experimental PredictionsExtractor implementation.

*   Stop returning empty example_ids metric from binary_confusion_matrices
    derived computations when example_id_key is not set but use_histrogam is
    true.

*   Add transformed features lookup for NDCG metrics query key and gain key.

*   Deprecate BoundedValue and TDistribution in ConfusionMatrixAtThresholds.

*   Fix a bug that dataframe auto_pivot fails if there is only Overall slice.

*   Use SavedModel PB to determine default signature instead of loading the
    model.

*   Reduce clutter in the multi-index columns and index in the experimental
    dataframe auto_pivot util.

*   Minor predictions_extractor_test.py refactor with readability improvements
    and improved test coverage.
*   Add an example of object detection metrics.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.41.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Move the version to top of init.py since the original "from
    tensorflow_model_analysis.sdk import *" will not import private symbol.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.41.0

## Major Features and Improvements

*   Add COCO object detection metrics, object detection related utilities,
    objection detection options in binary confusion matrix, Precision At Recall,
    and AUC. Add MaxRecall metric.
*   Add support for parsing sparse tensors with explicit tensor representations
    via TFXIO.

## Bug fixes and other Changes

*   Add score_distribution_plot.
*   Separate the Predictions Extractor into two extractors.
*   Update PredictionsExtractor to support backwards compatibility with the
    Materialized Predictions Extractor.
*   Depends on `apache-beam[gcp]>=2.40,<3`.
*   Depends on `pyarrow>=6,<7`.
*   Update merge_extracts with an option to skip squeezing one-dim arrays.
    Update split_extracts with an option to expand zero-dim arrays.
*   Added experimental bulk inference implementation to PredictionsExtractor.
    Currently only supports the RegressionAPI.

## Breaking Changes

*   Adds multi-index columns for view.experimental.metrics_as_dataframe util.
*   Changes SymmetricPredictionDifference output type from array to scalar.

## Deprecations

*   N/A

# Version 0.40.0

## Major Features and Improvements

*   Add object detection related utilities.

## Bug fixes and other Changes

*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.9,<3`
*   Fix issue where labels with -1 values are one-hot encoded when they
    shouldn't be ## Breaking Changes
*   Depends on `tfx-bsl>=1.9.0,<1.10.0`.
*   Depends on `tensorflow-metadata>=1.9.0,<1.10.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.39.0

## Major Features and Improvements

*   `SqlSliceKeyExtractor` now supports slicing on transformed features.

## Bug fixes and other Changes

*   Depends on `tfx-bsl>=1.8.0,<1.9.0`.
*   Depends on `tensorflow-metadata>=1.8.0,<1.9.0`.
*   Depends on `apache-beam[gcp]>=2.38,<3`.
*   Fix the incorrect keras.metrics.serialization for AUCPrecisionRecall.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.38.0

## Major Features and Improvements

*   Creates a VarLenTensorValue for storing batched, variable length extracts.
*   Adds a load_metrics_as_dataframe util to load metrics file as dataframe.

## Bug fixes and other Changes

*   Fixes issue attempting to parse metrics, plots, and attributions without a
    format suffix.

*   Fixes the non-deterministic key ordering caused by proto string
    serialization in metrics validator.

*   Update variable name to respectful terminology, rebuild JS

*   Fixes issues preventing standard preprocessors from being applied.

*   Allow merging extracts including sparse tensors with different dense shapes.

*   Allow counterfactual metrics to be calculated from predictions instead of
    only features.

## Breaking Changes

*   MetricsPlotsAndValidationsWriter will now write files with an explicit
    output format suffix (".tfrecord" by default). This should only affect
    pipelines which directly construct `MetricsPlotsAndValidationWriter`
    instances and do not set `output_file_format`. Those which use
    `default_writers()` should be unchanged.
*   Batched based extractors previously worked off of either lists of dicts of
    single tensor values or arrow record batches. These have been updated to be
    based on dicts with batched tensor values at the leaves.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on `tfx-bsl>=1.7.0,<1.8.0`.
*   Depends on `tensorflow-metadata>=1.7.0,<1.8.0`.
*   Depends on `apache-beam[gcp]>=2.36,<3`.

## Deprecations

*   N/A

# Version 0.37.0

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Fix Fairness Indicators UI bug with overlapping charts when comparing
    EvalResults
*   Fixed issue with aggregation type not being set properly in keys associated
    with confusion matrix metrics.
*   Enabled the sql_slice_key extractor when evaluating a model.
*   Depends on `numpy>=1.16,<2`.
*   Depends on `absl-py>=0.9,<2.0.0`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `tfx-bsl>=1.6.0,<1.7.0`.
*   Depends on `tensorflow-metadata>=1.6.0,<1.7.0`.
*   Depends on `apache-beam[gcp]>=2.35,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.36.0

## Major Features and Improvements

*   Replaced keras metrics with TFMA implementations. To use a keras metric in a
    `tfma.MetricConfig` you must now specify a module (i.e. `tf.keras.metrics`).
*   Added FixedSizeSample metric which can be used to extract a random,
    per-slice, fixed-sized sample of values for a user-configured feature key.

## Bug fixes and other Changes

*   Updated QueryStatistics to support weighted examples.
*   Replace confusion matrix based metrics with numpy counterparts, shifting
    away from Keras metrics class.
*   Depends on `apache-beam[gcp]>=2.34,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `tfx-bsl>=1.5.0,<1.6.0`.
*   Depends on `tensorflow-metadata>=1.5.0,<1.6.0`.

## Breaking Changes

*   Removes register_metric from public API, as it is not intended to be public
    facing. To use a custom metric, provide the module name in which the metric
    is defined in the MetricConfig message, instead.

## Deprecations

# Version 0.35.0

## Major Features and Improvements

*   Added support for specifying weighted vs unweighted metrics. The setting is
    available in the `tfma.MetricsSpec(
    example_weights=tfma.ExampleWeightOptions(weighted=True, unweighted=True))`.
    If no options are provided then TFMA will default to weighted provided the
    associated `tfma.ModelSpec` has an example weight key configured, otherwise
    unweighted will be used.

## Bug fixes and other Changes

*   Added support for example_weights that are arrays.

*   Reads baseUrl in JupyterLab to support TFMA rendering:
    https://github.com/tensorflow/model-analysis/issues/112

*   Fixing couple of issues with CIDerivedMetricComputation:

    *   no CI derived metric, deriving from private metrics such as
        binary_confusion_matrices, was being computed
    *   convert_slice_metrics_to_proto method didn't have support for bounded
        values metrics.

*   Depends on `tfx-bsl>=1.4.0,<1.5.0`.

*   Depends on `tensorflow-metadata>=1.4.0,<1.5.0`.

*   Depends on `apache-beam[gcp]>=2.33,<3`.

## Breaking Changes

*   Confidence intervals for scalar metrics are no longer stored in the
    `MetricValue.bounded_value`. Instead, the confidence interval for a metric
    can be found under `MetricKeysAndValues.confidence_interval`.
*   MetricKeys now require specifying whether they are weighted (
    `tfma.metrics.MetricKey(..., example_weighted=True)`) or unweighted (the
    default). If the weighting is unknown then `example_weighted` will be None.
    Any metric computed outside of a `tfma.metrics.MetricConfig` setting (i.e.
    metrics loaded from a saved model) will have the weighting set to None.
*   `ExampleCount` is now weighted based on `tfma.MetricsSpec.example_weights`
    settings. `WeightedExampleCount` has been deprecated (use `ExampleCount`
    instead). To get unweighted example counts (i.e. the previous implementation
    of `ExampleCount`), `ExampleCount` must now be added to a `MetricsSpec`
    where `example_weights.unweighted` is true. To get a weighted example count
    (i.e. what was previously `WeightedExampleCount`), `ExampleCount` must now
    be added to a `MetricsSpec` where `example_weights.weighted` is true.

## Deprecations

*   Deprecated python3.6 support.

# Version 0.34.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Correctly skips non-numeric numpy array type metrics for confidence interval
    computations.
*   Depends on `apache-beam[gcp]>=2.32,<3`.
*   Depends on `tfx-bsl>=1.3.0,<1.4.0`.

## Breaking Changes

*   In preparation for TFMA 1.0, the following imports have been moved (note
    that other modules were also moved, but TFMA only supports types that are
    explicitly declared inside of `__init__.py` files):
    *   `tfma.CombineFnWithModels` -> `tfma.utils.CombineFnWithModels`
    *   `tfma.DoFnWithModels` -> `tfma.utils.DoFnWithModels`
    *   `tfma.get_baseline_model_spec` -> `tfma.utils.get_baseline_model_spec`
    *   `tfma.get_model_type` -> `tfma.utils.get_model_type`
    *   `tfma.get_model_spec` -> `tfma.utils.get_model_spec`
    *   `tfma.get_non_baseline_model_specs` ->
        `tfma.utils.get_non_baseline_model_specs`
    *   `tfma.verify_eval_config` -> `tfma.utils.verify_eval_config`
    *   `tfma.update_eval_config_with_defaults` ->
        `tfma.utils.update_eval_config_with_defaults`
    *   `tfma.verify_and_update_eval_shared_models` ->
        `tfma.utils.verify_and_update_eval_shared_models`
    *   `tfma.create_keys_key` -> `tfma.utils.create_keys_key`
    *   `tfma.create_values_key` -> `tfma.utils.create_values_key`
    *   `tfma.compound_key` -> `tfma.utils.compound_key`
    *   `tfma.unique_key` -> `tfma.utils.unique_key`

## Deprecations

*   N/A

# Version 0.34.0

## Major Features and Improvements

*   Added `SparseTensorValue` and `RaggedTensorValue` types for storing
    in-memory versions of sparse and ragged tensor values used in extracts.
    Tensor values used for features, etc should now be based on either
    `np.ndarray`, `SparseTensorValue`, or `RaggedTensorValue` and not
    tf.compat.v1 value types.
*   Add `CIDerivedMetricComputation` metric type.

## Bug fixes and other Changes

*   Depends on `pyarrow>=1,<6`.
*   Fixes bug when computing confidence intervals for
    `binary_confusion_metrics.ConfusionMatrixAtThresholds` (or any other
    structured metric).
*   Fixed bug where example_count post_export_metric is added even if
    include_default_metrics is False.
*   Depends on `apache-beam[gcp]>=2.31,<2.32`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3`.
*   Depends on `tfx-bsl>=1.3.1,<1.4.0`.
*   Fixes issue with jackknife confidence interval method that resulted in
    erroneously large intervals.
*   Fixes bug where calls to `_apply_binary_op_elementwise` could fail on
    objects of types `binary_confusion_matrices.Matrices` and
    `multi_class_confusion_matrix_metrics.Matrices` due to differing thresholds.

## Breaking Changes

*   Missing baseline model when change thresholds are present is not allowed
    anymore, an exception will be raised unless the rubber_stamp flag is True.

## Deprecations

*   N/A

# Version 0.33.0

## Major Features and Improvements

*   Provided functionality for `slice_keys_sql` config. It's not available under
    Windows.
*   The `confidence_interval` field within `metrics_for_slice_pb2.MetricValue`
    has been removed and the tag number reserved. This information now lives in
    `metrics_for_slice_pb2.MetricKeyAndValue.confidence_interval`.

## Bug fixes and other Changes

*   Improve rendering of HTML stubs for TFMA and Fairness Indicators UI.
*   Update README for JupyterLab 3
*   Provide implementation of ExactMatch metric.
*   Jackknife CI method now works with cross-slice metrics.
*   Depends on `apache-beam[gcp]>=2.31,<3`.
*   Depends on `tensorflow-metadata>=1.2.0,<1.3.0`.
*   Depends on `tfx-bsl>=1.2.0,<1.3.0`.

## Breaking Changes

*   The binary_confusion_matrices metric formerly returned confusion matrix
    counts (i.e number of {true,false} {positives,negatives}) and optionally a
    set of representative examples in a single object. Now, this metric class
    generates two separate metrics values when examples are configured: one
    containing just the counts, and the other just examples. This should only
    affect users who created a custom derived metric that used
    binary_confusion_matrices metric as an input.

## Deprecations

*   N/A

# Version 0.32.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Depends on `google-cloud-bigquery>>=1.28.0,<2.21`.
*   Depends on `tfx-bsl>=1.1.0,<1.2.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.32.0

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Depends on `protobuf>=3.13,<4`.
*   Depends on `tensorflow-metadata>=1.1.0,<1.2.0`.
*   Depends on `tfx-bsl>=1.1.0,<1.2.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.31.0

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Depends on `apache-beam[gcp]>=2.29,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<3`.
*   Depends on `tensorflowjs>=3.6.0,<4`.
*   Depends on `tensorflow-metadata>=1.0.0,<1.1.0`.
*   Depends on `tfx-bsl>=1.0.0,<1.1.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.30.0

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Fix bug that `FeaturesExtractor` incorrectly handles RecordBatches that have
    only the raw input column but no other feature columns.

*   Fix an issue that micro_average can get lost in MetricKey, which can cause
    threshold mismatch the metrics during validation.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.29.0

## Major Features and Improvements

*   Added support for output aggregation.

## Bug fixes and other Changes

*   In Fairness Indicators UI, sort metrics list to show common metrics first
*   For lift metrics, support negative values in the Fairness Indicator UI bar
    chart.
*   Make legacy predict extractor also input/output batched extracts.
*   Updated to use new compiled_metrics and compiled_loss APIs for keras
    in-graph metric computations.
*   Add support for calling model.evaluate on keras models containing custom
    metrics.
*   Add CrossSliceMetricComputation metric type.
*   Add Lift metrics under addons/fairness.
*   Don't add metric config from config.MetricsSpec to baseline model spec by
    default.
*   Fix invalid calculations for metrics derived from tf.keras.losses.
*   Fixes following bugs related to CrossSlicingSpec based evaluation results.
    *   metrics_plots_and_validations_writer was failing while writing cross
        slice comparison results to metrics file.
    *   Fairness widget view was not compatible with cross slicing key type.
*   Fix support for exporting the UI from a notebook to a standalone HTML page.
*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `tensorflow-metadata>=0.29.0,<0.30.0`.
*   Depends on `tfx-bsl>=0.29.0,<0.30.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.28.0

## Major Features and Improvements

*   Add a new base computation for binary confusion matrix (other than based on
    calibration histogram). It also provides a sample of examples for the
    confusion matrix.
*   Adding two new metrics - Flip Count and Flip Rate to evaluate Counterfactual
    Fairness.

## Bug fixes and other Changes

*   Fixed division by zero error for diff metrics.
*   Depends on `apache-beam[gcp]>=2.28,<3`.
*   Depends on `numpy>=1.16,<1.20`.
*   Depends on `tensorflow-metadata>=0.28.0,<0.29.0`.
*   Depends on `tfx-bsl>=0.28.0,<0.29.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.27.0

## Major Features and Improvements

*   Created tfma.StandardExtracts with helper methods for common keys.
*   Updated StandardMetricInputs to extend from the tfma.StandardExtracts.
*   Created set of StandardMetricInputsPreprocessors for filtering extracts.
*   Introduced a `padding_options` config to ModelSpec to configure whether and
    how to pad the prediction and label tensors expected by the model's metrics.

## Bug fixes and other changes

*   Fixed issue with metric computation deduplication logic.
*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-metadata>=0.27.0,<0.28.0`.
*   Depends on `tfx-bsl>=0.27.0,<0.28.0`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.26.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Fix support for exporting the UI from a notebook to a standalone HTML page.
*   Depends on apache-beam[gcp]>=2.25,!=2.26,<2.29.
*   Depends on numpy>=1.16,<1.20.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.26.0

## Major Features and Improvements

*   Added support for aggregating feature attributions using special metrics
    that extend from `tfma.metrics.AttributionMetric` (e.g.
    `tfma.metrics.TotalAttributions`, `tfma.metrics.TotalAbsoluteAttributions`).
    To use make use of these metrics a custom extractor that add attributions to
    the `tfma.Extracts` under the key name `tfma.ATTRIBUTIONS_KEY` must be
    manually created.
*   Added support for feature transformations using TFT and other preprocessing
    functions.
*   Add support for rubber stamping (first run without a valid baseline model)
    when validating a model. The change threshold is ignored only when the model
    is rubber stamped, otherwise, an error is thrown.

## Bug fixes and other changes

*   Fix the bug that Fairness Indicator UI metric list won't refresh if the
    input eval result changed.
*   Add support for missing_thresholds failure to validations result.
*   Updated to set min/max value for precision/recall plot to 0 and 1.
*   Fix issue with MinLabelPosition not being sorted by predictions.
*   Updated NDCG to ignore non-positive gains.
*   Fix bug where an example could be aggregated more than once in a single
    slice if the same slice key were generated from more than one SlicingSpec.
*   Add threshold support for confidence interval type metrics based on its
    unsampled_value.
*   Depends on `apache-beam[gcp]>=2.25,!=2.26.*,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-metadata>=0.26.0,<0.27.0`.
*   Depends on `tfx-bsl>=0.26.0,<0.27.0`.

## Breaking changes

*   Changed MultiClassConfusionMatrix threshold check to prediction > threshold
    instead of prediction >= threshold.
*   Changed default handling of materialize in default_extractors to False.
*   Separated `tfma.extractors.BatchedInputExtractor` into
    `tfma.extractors.FeaturesExtractor`, `tfma.extractors.LabelsExtractor`, and
    `tfma.extractors.ExampleWeightsExtractor`.
*   Change the thresholding to be inclusive, i.e. model passes when value is >=
    or <= to the threshold rather than > or <.

## Deprecations

*   N/A

# Version 0.25.0

## Major Features and Improvements

*   Added support for reading and writing metrics, plots and validation results
    using Apache Parquet.
*   Updated the FI indicator slicing selection UI.
*   Fixed the problem that slices are refreshed when user selected a new
    baseline.
*   Add support for slicing on ragged and multidimensional data.
*   Load TFMA correctly in JupyterLabs even if Facets has loaded first.
*   Added support for aggregating metrics using top k values.
*   Added support for padding labels and predictions with -1 to align a batch of
    inputs for use in tf-ranking metrics computations.
*   Added support for fractional labels.
*   Add metric definitions as tooltips in the Fairness Inidicators metric
    selector UI
*   Added support for specifying label_key to use with MinLabelPosition metric.
*   From this release TFMA will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple tensorflow-model-analysis
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of TFMA available on PyPI by running the
    command `pip install tensorflow-model-analysis`.

## Bug fixes and other changes

*   Fix incorrect calculation with MinLabelPosition when used with weighted
    examples.
*   Fix issue with using NDCG metric without binarization settings.
*   Fix incorrect computation when example weight is set to zero.
*   Depends on `apache-beam[gcp]>=2.25,<3`.
*   Depends on `tensorflow-metadata>=0.25.0,<0.26.0`.
*   Depends on `tfx-bsl>=0.25.0,<0.26.0`.

## Breaking changes

*   `AggregationOptions` are now independent of `BinarizeOptions`. In order to
    compute `AggregationOptions.macro_average` or
    `AggregationOptions.weighted_macro_average`,
    `AggregationOptions.class_weights` must now be configured. If
    `AggregationOptions.class_weights` are provided, any missing keys now
    default to 0.0 instead of 1.0.
*   In the UI, aggregation based metrics will now be prefixed with 'micro_',
    'macro_', or 'weighted_macro_' depending on the aggregation type.

## Deprecations

*   `tfma.extractors.FeatureExtractor`, `tfma.extractors.PredictExtractor`,
    `tfma.extractors.InputExtractor`, and
    `tfma.evaluators.MetricsAndPlotsEvaluator` are deprecated and may be
    replaced with newer versions in upcoming releases.

# Version 0.24.3

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Depends on `apache-beam[gcp]>=2.24,<3`.
*   Depends on `tfx-bsl>=0.24.0,<0.25.0`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.24.2

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Added an extra requirement group `all`. As a result, barebone TFMA does not
    require `tensorflowjs` , `prompt-toolkit` and `ipython` any more.
*   Added an extra requirement group `all` that specifies all the extra
    dependencies TFMA needs. Use `pip install tensorflow-model-analysis[all]` to
    pull in those dependencies.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.24.1

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Fix Jupyter lab issue with missing data-base-url.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.24.0

## Major Features and Improvements

*   Use TFXIO and batched extractors by default in TFMA.

## Bug fixes and other changes

*   Updated the type hint of FilterOutSlices.
*   Fix issue with precision@k and recall@k giving incorrect values when
    negative thresholds are used (i.e. keras defaults).
*   Fix issue with MultiClassConfusionMatrixPlot being overridden by
    MultiClassConfusionMatrix metrics.
*   Made the Fairness Indicators UI thresholds drop down list sorted.
*   Fix the bug that Sort menu is not hidden when there is no model comparison.
*   Depends on `absl-py>=0.9,<0.11`.
*   Depends on `ipython>=7,<8`.
*   Depends on `pandas>=1.0,<2`.
*   Depends on `protobuf>=3.9.2,<4`.
*   Depends on `tensorflow-metadata>=0.24.0,<0.25.0`.
*   Depends on `tfx-bsl>=0.24.0,<0.25.0`.

## Breaking changes

*   Query based metrics evaluations that make use of `MetricsSpecs.query_key`
    are now passed `tfma.Extracts` with leaf values that are of type
    `np.ndarray` containing an additional dimension representing the values
    matched by the query (e.g. if the labels and predictions were previously 1D
    arrays, they will now be 2D arrays where the first dimension's size is equal
    to the number of examples matching the query key). Previously a list of
    `tfma.Extracts` was passed instead. This allows user's to now add custom
    metrics based on `tf.keras.metrics.Metric` as well as `tf.metrics.Metric`
    (any previous customizations based on `tf.metrics.Metric` will need to be
    updated). As part of this change the `tfma.metrics.NDCG`,
    `tfma.metrics.MinValuePosition`, and `tfma.metrics.QueryStatistics` have
    been updated.
*   Renamed `ConfusionMatrixMetric.compute` to `ConfusionMatrixMetric.result`
    for consistency with other APIs.

## Deprecations

*   Deprecating Py3.5 support.

# Version 0.23.0

## Major Features and Improvements

*   Changed default confidence interval method from POISSON_BOOTSTRAP to
    JACKKNIFE. This should significantly improve confidence interval evaluation
    performance by as much as 10x in runtime and CPU resource usage.
*   Added support for additional confusion matrix metrics (FDR, FOR, PT, TS, BA,
    F1 score, MCC, FM, Informedness, Markedness, etc). See
    https://en.wikipedia.org/wiki/Confusion_matrix for full list of metrics now
    supported.
*   Change the number of partitions used by the JACKKNIFE confidence interval
    methodology from 100 to 20. This will reduce the quality of the confidence
    intervals but support computing confidence intervals on slices with fewer
    examples.
*   Added `tfma.metrics.MultiClassConfusionMatrixAtThresholds`.
*   Refactoring code to compute `tfma.metrics.MultiClassConfusionMatrixPlot`
    using derived computations.
*   Provide support for evaluating TFJS models.

## Bug fixes and other changes

*   Added support for labels passed as SparseTensorValues.
*   Stopped requiring `avro-python3`.
*   Fix NoneType error when passing BinarizeOptions to
    tfma.metrics.default_multi_class_classification_specs.
*   Fix issue with custom metrics contained in modules ending in
    tf.keras.metric.
*   Changed the BoundedValue.value to be the unsampled metric value rather than
    the sample average.
*   Add `EvalResult.get_metric_names()`.
*   Added errors for missing slices during metrics validation.
*   Added support for customizing confusion matrix based metrics in keras.
*   Made BatchedInputExtractor externally visible.
*   Updated tfma.load_eval_results API to return empty results instead of
    throwing an error when evaluation results are missing for a model_name.
*   Fixed an issue in Fairness Indicators UI where omitted slices error message
    was being displayed even if no slice was omitted.
*   Fix issue with slice_spec.is_slice_applicable not working for float, int,
    etc types that are encoded as strings.
*   Wrap long strings in table cells in Fairness Indicators UI.
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `scipy>=1.4.1,<2`
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
*   Depends on `tensorflow-metadata>=0.23,<0.24`.
*   Depends on `tfx-bsl>=0.23,<0.24`.

## Breaking changes

*   Rename EvalResult.get_slices() to EvalResult.get_slice_names().

## Deprecations

*   Note: We plan to remove Python 3.5 support after this release.

# Version 0.22.2

## Major Features and Improvements

*   Added analyze_raw_data(), an API for evaluating TFMA metrics on Pandas
    DataFrames.

## Bug fixes and other changes

*   Previously metrics would only be computed for combinations of keys that
    produced different metric values (e.g. `ExampleCount` will be the same for
    all models, outputs, classes, etc, so only one metric key was used). Now a
    metric key will be returned for each combination associated with the
    `MetricSpec` definition even if the values will be the same. Support for
    model independent metrics has also been removed. This means by default
    multiple ExampleCount metrics will be created when multiple models are used
    (one per model).
*   Fixed issue with label_key and prediction_key settings not working with TF
    based metrics.
*   Fairness Indicators UI
    *   Thresholds are now sorted in ascending order.
    *   Barchart can now be sorted by either slice or eval.
*   Added support for slicing on any value extracted from the inputs (e.g. raw
    labels).
*   Added support for filtering extracts based on sub-keys.
*   Added beam counters to track the feature slices being used for evaluation.
*   Adding KeyError when analyze_raw_data is run without a valid label_key or
    prediction_key within the provided Pandas DataFrame.
*   Added documentation for `tfma.analyze_raw_data`, `tfma.view.SlicedMetrics`,
    and `tfma.view.SlicedPlots`.
*   Unchecked Metric thresholds now block the model validation.
*   Added support for per slice threshold settings.
*   Added support for sharding metrics and plots outputs.
*   Updated load_eval_result to support filtering plots by model name. Added
    support for loading multiple models at same output path using
    load_eval_results.
*   Fix typo in jupyter widgets breaking TimeSeriesView and PlotViewer.
*   Add `tfma.slicer.stringify_slice_key()`.
*   Deprecated external use of tfma.slicer.SingleSliceSpec (tfma.SlicingSpec
    should be used instead).
*   Updated tfma.default_eval_shared_model and tfma.default_extractors to better
    support custom model types.
*   Depends on 'tensorflow-metadata>=0.22.2,<0.23'

## Breaking changes

*   Changed to treat CLASSIFY_OUTPUT_SCORES involving 2 values as a multi-class
    classification prediction instead of converting to binary classification.
*   Refactored confidence interval methodology field. The old path under
    `Options.confidence_interval_methodology` is now at
    `Options.confidence_intervals.methodology`.
*   Removed model_load_time_callback from ModelLoader construct_fn (timing is
    now handled by load). Removed access to shared_handle from ModelLoader.

## Deprecations

# Version 0.22.1

## Major Features and Improvements

## Bug fixes and other changes

*   Depends on `pyarrow>=0.16,<0.17`.

## Breaking changes

## Deprecations

# Version 0.22.0

## Major Features and Improvements

*   Added support for jackknife-based confidence intervals.
*   Add EvalResult.get_metrics(), which extracts slice metrics in dictionary
    format from EvalResults.
*   Adds TFMD `Schema` as an available argument to computations callbacks.

## Bug fixes and other changes

*   Version is now available under `tfma.version.VERSION` or `tfma.__version__`.
*   Add auto slicing utilities for significance testing.
*   Fixed error when a metric and loss with the same classname are used.
*   Adding two new ratios (false discovery rate and false omission rate) in
    Fairness Indicators.
*   `MetricValue`s can now contain both a debug message and a value (rather than
    one or the other).
*   Fix issue with displaying ConfusionMatrixPlot in colab.
*   `CalibrationPlot` now infers `left` and `right` values from schema, when
    available. This makes the calibration plot useful to regression users.
*   Fix issue with metrics not being computed properly when mixed with specs
    containing micro-aggregation computations.
*   Remove batched keys. Instead use the same keys for batched and unbatched
    extract.
*   Adding support to visualize Fairness Indicators in Fairness Indicators
    TensorBoard Plugin by providing remote evalution path in query parameter:
    `<tensorboard_url>#fairness_indicators&
    p.fairness_indicators.evaluation_output_path=<evaluation_path>`.
*   Fixed invalid metrics calculations for serving models using the
    classification API with binary outputs.
*   Moved config writing code to extend from tfma.writer.Writer and made it a
    member of default_writers.
*   Updated tfma.ExtractEvaluateAndWriteResults to accept Extracts as input in
    addition to serialize bytes and Arrow RecordBatches.
*   Depends on `apache-beam[gcp]>=2.20,<3`.
*   Depends on `pyarrow>=0.16,<1`.
*   Depends on `tensorflow>=1.15,!=2.0.*,<3`.
*   Depends on `tensorflow-metadata>=0.22,<0.23`.
*   Depends on `tfx-bsl>=0.22,<0.23`.

## Breaking changes

*   Remove desired_batch_size as an option. Large batch failures can be handled
    via serially processing the failed batch which also acts as a deterent from
    scaling up batch sizes further. Batch size can be handled via BEAM batch
    size tuning.

## Deprecations

*   Deprecating Py2 support.

# Release 0.21.6

## Major Features and Improvements

*   Integrate TFXIO in TFMA. Use batched input and predict extractor in V2.
    Results in ~40% reduction in CPU seconds over existing TFMA v2
    (InputExtractor + PredictExtractorV2). Modify TFMA public API to take an
    optional tensor adapter config as input.
*   Adding experimental support for pre-defined preprocessing functions that can
    be used as preprocessing functions for feature and label transformations.

## Bug fixes and other changes

*   Populate confidence_interval field in addition to bounded_value when
    confidence intervals is enabled.
*   Only requires `avro-python3>=1.8.1,!=1.9.2.*,<2.0.0` on Python 3.5 + MacOS
*   Fix bug in SensitivitySpecificityBase derived metrics: guarantee well
    defined behaviour when the constraint lies between feasible points (see
    updated docstrings).

## Breaking changes

## Deprecations

# Release 0.21.5

## Major Features and Improvements

*   Now publish NPM under `tensorflow_model_analysis` for UI components.

## Bug fixes and other changes

*   Depends on 'tfx-bsl>=0.21.3,<0.22',
*   Depends on 'tensorflow>=1.15,<3',
*   Depends on 'apache-beam[gcp]>=2.17,<3',

## Breaking changes

*   Rollback populating TDistributionValue metric when confidence intervals is
    enabled in V2.
*   Drop Py2 support.

## Deprecations

# Release 0.21.4

## Major Features and Improvements

*   Added support for creating metrics specs from tf.keras.losses.
*   Added evaluation comparison feature to the Fairness Indicators UI in Colab.
*   Added better defaults handling for eval config so that a single model spec
    can be used for both candidate and baseline.
*   Added support to provide output file format in load_eval_result API.

## Bug fixes and other changes

*   Fixed issue with keras metrics saved with the model not being calculated
    unless a keras metric was added to the config.
*   Depends on `pandas>=0.24,<2`.
*   Depends on `pyarrow>=0.15,<1`.
*   Depends on 'tfx-bsl>=0.21.3,<0.23',
*   Depends on 'tensorflow>=1.15,!=2.0.*,<3',
*   Depends on 'apache-beam[gcp]>=2.17,<2.18',

## Deprecations

# Release 0.21.3

## Major Features and Improvements

*   Added support for model validation using either value threshold or diff
    threshold.
*   Added a writer to output model validation result (ValidationResult).
*   Added support for multi-model evaluation using EvalSavedModels.
*   Added support for inserting model_names by default to metrics_specs.
*   Added support for selecting custom model format evals in config.

## Bug fixes and other changes

*   Fixed issue with model_name not being set in keras metrics.

## Breaking changes

*   Populate TDistributionValue metric when confidence intervals is enabled in
    V2.
*   Rename the writer MetricsAndPlotsWriter to MetricsPlotsAndValidationsWriter.

## Deprecations

# Release 0.21.2

## Major Features and Improvements

## Bug fixes and other changes

*   Adding SciPy dependency for both Python2 and Python3
*   Increased table and tooltip font in Fairness Indicators.

## Breaking changes

*   `tfma.BinarizeOptions.class_ids`, `tfma.BinarizeOptions.k_list`,
    `tfma.BinarizeOptions.top_k_list`, and `tfma.Options.disabled_outputs` are
    now wrapped in an additional proto message.

## Deprecations

# Release 0.21.1

## Major Features and Improvements

*   Adding a TFLite predict extractor to enable obtaining inferences from TFLite
    models.

## Bug fixes and other changes

*   Adding support to compute deterministic confidence intervals using a seed
    value in tfma.run_model_analysis API for testing or experimental purposes.
*   Fixed calculation of `tfma.metrics.CoefficientOfDiscrimination` and
    `tfma.metrics.RelativeCoefficientOfDiscrimination`.

## Breaking changes

*   Renaming k_anonymization_count field name to min_slice_size.

## Deprecations

# Release 0.21.0

## Major Features and Improvements

*   Added `tfma.metrics.MinLabelPosition` and `tfma.metrics.QueryStatistics` for
    use with V2 metrics API.
*   Added `tfma.metrics.CoefficientOfDiscrimination` and
    `tfma.metrics.RelativeCoefficientOfDiscrimination` for use with V2 metrics
    API.
*   Added support for using `tf.keras.metrics.*` metrics with V2 metrics API.
*   Added support for default V2 MetricSpecs and creating specs from
    `tf.kera.metrics.*` and `tfma.metrics.*` metric classes.
*   Added new MetricsAndPlotsEvaluator based on V2 infrastructure. Note this
    evaluator also supports query-based metrics.
*   Add support for micro_average, macro_average, and weighted_macro_average
    metrics.
*   Added support for running V2 extractors and evaluators. V2 extractors will
    be used whenever the default_eval_saved_model is created using a non-eval
    tag (e.g. `tf.saved_model.SERVING`). The V2 evaluator will be used whenever
    a `tfma.EvalConfig` is used containing `metrics_specs`.
*   Added support for `tfma.metrics.SquaredPearsonCorrelation` for use with V2
    metrics API.
*   Improved support for TPU autoscaling and handling batch_size related
    scaling.
*   Added support for `tfma.metrics.Specificity`, `tfma.metrics.FallOut`, and
    `tfma.metrics.MissRate` for use with V2 metrics API. Renamed `AUCPlot` to
    `ConfusionMatrixPlot`, `MultiClassConfusionMatrixAtThresholds` to
    `MultiClassConfusionMatrixPlot` and `MultiLabelConfusionMatrixAtThresholds`
    to `MultiLabelConfusionMatrixPlot`.
*   Added Jupyter support to Fairness Indicators. Currently does not support WIT
    integration.
*   Added fairness indicators metrics
    `tfma.addons.fairness.metrics.FairnessIndicators`.
*   Updated documentation for new metrics infrastructure and newly supported
    models (keras, etc).
*   Added support for model diff metrics. Users need to turn on "is_baseline" in
    the corresponding ModelSpec.

## Bug fixes and other changes

*   Fixed error in `tfma-multi-class-confusion-matrix-at-thresholds` with
    default classNames value.
*   Fairness Indicators
    -   Compute ratio metrics with safe division.
    -   Remove "post_export_metrics" from metric names.
    -   Move threshold dropdown selector to a metric-by-metric basis, allowing
        different metrics to be inspected with different thresholds. Don't show
        thresholds for metrics that do not support them.
    -   Slices are now displayed in alphabetic order.
    -   Adding an option to "Select all" metrics in UI.
*   Added auto slice key extractor based on statistics.
*   Depends on 'tensorflow-metadata>=0.21,<0.22'.
*   Made InputProcessor externally visible.

## Breaking changes

*   Updated proto config to remove input/output data specs in favor of passing
    them directly to the run_eval.

## Deprecations

# Release 0.15.4

## Major Features and Improvements

## Bug fixes and other changes

*   Fixed the bug that Fairness Indicator will skip metrics with NaN value.

## Breaking changes

## Deprecations

# Release 0.15.3

## Major Features and Improvements

## Bug fixes and other changes

*   Updated vulcanized_tfma.js with UI changes in addons/fairness_indicators.

## Breaking changes

## Deprecations

# Release 0.15.2

## Major Features and Improvements

## Bug fixes and other changes

*   Updated to use tf.io.gfile for reading config files (fixes issue with
    reading from GCS/HDFS in 0.15.0 and 0.15.1 releases).

## Breaking changes

## Deprecations

# Release 0.15.1

## Major Features and Improvements

*   Added support for defaulting to using class IDs when classes are not present
    in outputs for multi-class metrics (for use in keras model_to_estimator).
*   Added example count metrics (`tfma.metrics.ExampleCount` and
    `tfma.metrics.WeightedExampleCount`) for use with V2 metrics API.
*   Added calibration metrics (`tfma.metrics.MeanLabel`,
    `tfma.metrics.MeanPrediction`, and `tfma.metrics.Calibration`) for use with
    V2 metrics API.
*   Added `tfma.metrics.ConfusionMatrixAtThresholds` for use with V2 metrics
    API.
*   Added `tfma.metrics.CalibrationPlot` and `tfma.metrics.AUCPlot` for use with
    V2 metrics API.
*   Added multi_class / multi_label plots (
    `tfma.metrics.MultiClassConfusionMatrixAtThresholds`,
    `tfma.metrics.MultiLabelConfusionMatrixAtThresholds`) for use with V2
    metrics API.
*   Added `tfma.metrics.NDCG` metric for use with V2 metrics API.
*   Added `calibration` as a post export metric.

## Bug fixes and other changes

*   Depends on `tensorflow>=1.15,<3.0`.
    *   Starting from 1.15, package `tensorflow` comes with GPU support. Users
        won't need to choose between `tensorflow` and `tensorflow-gpu`.
    *   Caveat: `tensorflow` 2.0.0 is an exception and does not have GPU
        support. If `tensorflow-gpu` 2.0.0 is installed before installing
        `tensorflow_model_analysis`, it will be replaced with `tensorflow`
        2.0.0. Re-install `tensorflow-gpu` 2.0.0 if needed.

## Breaking changes

## Deprecations

# Release 0.15.0

## Major Features and Improvements

*   Added V2 of PredictExtractor that uses TF 2.0 signature APIs and supports
    keras models (note: keras model evaluation not fully supported yet).
*   `tfma.run_model_analysis`, `tfma.default_extractors`,
    `tfma.default_evaluators`, and `tfma.default_writers` now allow settings to
    be passed as an `EvalConfig`.
*   `tfma.run_model_analysis`, `tfma.default_extractors`,
    `tfma.default_evaluators`, and `tfma.default_writers` now allow multiple
    models to be passed (note: multi-model support not fully implemented yet).
*   Added InputExtractor for extracting labels, features, and example weights
    from tf.Examples.
*   Added Fairness Indicator as an addon.

## Bug fixes and other changes

*   Enabled TF 2.0 support using compat.v1.
*   Added support for slicing on native dicts of features in addition to FPL
    types.
*   For multi-output and / or multi-class models, please provide output_name and
    / or class_id to tfma.view.render_plot.
*   Replaced dependency on `tensorflow-transform` with `tfx-bsl`. If running
    with latest main, `tfx-bsl` must also be latest main.
*   Depends on `tfx-bsl>=0.15,<0.16`.
*   Slicing now supports conversion between int/floats and strings.
*   Depends on `apache-beam[gcp]>=2.16,<3`.
*   Depends on `six==1.12`.

## Breaking changes

*   tfma.EvalResult.slicing_metrics now contains nested dictionaries of output,
    class id and then metric names.
*   Update config serialization to use JSON instead of pickling and reformat
    config to include input_data_specs, model_specs, output_data_specs, and
    metrics_specs.
*   Requires pre-installed TensorFlow >=1.15,<3.

## Deprecations

# Release 0.14.0

## Major Features and Improvements

*   Added documentation on architecture.
*   Added an `adapt_to_remove_metrics` function to `tfma.exporter` which can be
    used to remove metrics incompatible with TFMA (e.g. `py_func` or streaming
    metrics) before exporting the TFMA EvalSavedModel.
*   Added support for passing sparse int64 tensors to precision/recall@k.
*   Added support for binarization of multiclass metrics that use labels of the
    from (N) in addition to (N, 1).
*   Added support for using iterators with EvalInputReceiver.
*   Improved performance of confidence interval computations by modifying the
    pipeline shape.
*   Added QueryBasedMetricsEvaluator which supports computing query-based
    metrics (e.g. normalized discounted cumulative gain).
*   Added support for merging metrics produced by different evaluators.
*   Added support for blocklisting specified features from fetches.
*   Added functionality to the FeatureExtractor to specify the features dict as
    a possible destination.
*   Added support for label vocabularies for binary and multi-class estimators
    that support the new ALL_CLASSES prediction output.
*   Move example parsing in aggregation into the graph for performance
    improvements in both standard and model_agnostic evaluation modes.
*   Created separate ModelLoader type for loading the EvalSavedModel.

## Bug fixes and other changes

*   Upgraded codebase for TF 2.0 compatibility.
*   Make metrics-related operations thread-safe by wrapping them with locks.
    This eliminates race conditions that were previously possible in
    multi-threaded runners which could result in incorrect metric values.
*   More flexible `FanoutSlices`.
*   Limit the number of sampling buckets to 20.
*   Improved performance in Confidence Interval computation.
*   Refactored poisson bootstrap code to be re-usable in other evaluators.
*   Refactored k-anonymity code to be re-usable in other evaluators.
*   Fixed slicer feature string value handling in Python3.
*   Added support for example weight keys for multi-output models.
*   Added option to set the desired batch size when calling run_model_analysis.
*   Changed TFRecord compression type from UNCOMPRESSED to AUTO.
*   Depends on `apache-beam[gcp]>=2.14,<3`.
*   Depends on `numpy>=1.16,<2`.
*   Depends on `protobuf>=3.7,<4`.
*   Depends on `scipy==1.1.0`.
*   Added support to change k_anonymization_count value via EvalConfig.

## Breaking changes

*   Removed uses of deprecated tf.contrib packages (where possible).
*   `tfma.default_writers` now requires the `eval_saved_model` to be passed as
    an argument.
*   Requires pre-installed TensorFlow >=1.14,<2.

## Deprecations

# Release 0.13.1

## Major Features and Improvements

*   Added support for squared pearson correlation (R squared) post export
    metric.
*   Added support for mean absolute error post export metric.
*   Added support for mean squared error and root mean squared error post export
    metric.
*   Added support for not computing metrics for slices with less than a given
    number of examples.

## Bug fixes and other changes

*   Cast / convert labels for precision / recall at K so that they work even if
    the label and the classes Tensors have different types, as long as the types
    are compatible.
*   Post export metrics will now also search for prediction keys prefixed by
    metric_tag if it is specified.
*   Added support for precision/recall @ k using canned estimators provided
    label vocab not used.
*   Preserve unicode type of slice keys when serialising to and deserialising
    from disk, instead of always converting them to bytes.
*   Use `__slots__` in accumulators.

## Breaking changes

*   Expose Python 3 types in the code (this will break Python 2 compatibility)

## Deprecations

# Release 0.13.0

## Major Features and Improvements

*   Python 3.5 is supported.

## Bug fixes and other changes

*   Added support for fetching additional tensors at prediction time besides
    features, predictions, and labels (predict now returns FetchedTensorValues
    type).
*   Removed internal usages of encoding.NODE_SUFFIX indirection within dicts in
    the eval_saved_model module (encoding.NODE_SUFFIX is still used in
    FeaturesPredictionLabels)
*   Predictions are now returned as tensors (vs dicts) when "predictions" is the
    only output (this is consistent with how features and labels work).
*   Depends on `apache-beam[gcp]>=2.11,<3`.
*   Depends on `protobuf>=3.7,<4`.
*   Depends on `scipy==1.1.0`.
*   Add support for multiple plots in a single evaluation.
*   Add support for changeable confidence levels.

## Breaking changes

*   Post export metrics for precision_recall_at_k were split into separate
    fuctions: precision_at_k and recall_at_k.
*   Requires pre-installed TensorFlow >=1.13,<2.

## Deprecations

# Release 0.12.0

## Major Features and Improvements

*   Python 3.5 readiness complete (all tests pass). Full Python 3.5
    compatibility is expected to be available with the next version of Model
    Analysis (after Apache Beam 2.11 is released).
*   Added support for customizing the pipeline (via extractors, evaluators, and
    writers). See [architecture](g3doc/architecture.md) for more details.
*   Added support for excluding the default metrics from the saved model graph
    during evaluation.
*   Added a mechanism for performing evaluations via post_export_metrics without
    access to a Tensorflow EvalSavedModel.
*   Added support for computing metrics with confidence intervals using the
    [Poisson bootstrap technique](http://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html).
    To use, set the num_bootstrap_samples to a number greater than 1--20 is
    recommended for confidence intervals.

## Bug fixes and other changes

*   Fixed bugs where TFMA was incorrectly modifying elements in DoFns, which
    violates the Beam API.
*   Fixed correctness issue stemming from TFMA incorrectly relying on evaluation
    ordering that TF doesn't guarantee.
*   We now store feature and label Tensor information in SignatureDef inputs
    instead of Collections in anticipation of Collections being deprecated in TF
    2.0.
*   Add support for fractional labels in AUC, AUPRC and confusion matrix at
    thresholds. Previously the labels were being passed directly to TensorFlow,
    which would cast them to `bool`, which meant that all non-zero labels were
    treated as positive examples. Now we treat a fractional label `l` in `[0,
    1]` as two examples, a positive example with weight `l` and a negative
    example with weight `1 - l`.
*   Depends on `numpy>=1.14.5,<2`.
*   Depends on `scipy==0.19.1`.
*   Depends on `protobuf==3.7.0rc2`.
*   Chicago Taxi example is moved to tfx repo
    (https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi)

## Breaking changes

*   Moved tfma.SingleSliceSpec to tfma.slicer.SingleSliceSpec.

## Deprecations

# Release 0.11.0

## Major Features and Improvements

*   We now support unsupervised models which have `model_fn`s that do not take a
    `labels` argument.
*   Improved performance by using `make_callable` instead of repeated
    `session.run` calls.
*   Improved performance by better choice of default "combine" batch size.
*   We now support passing in custom extractors in the model_eval_lib API.
*   Added support for models which have multiple examples per raw input (e.g.
    input is a compressed example which expands to multiple examples when parsed
    by the model). For such models, you must specify an `example_ref` parameter
    to your `EvalInputReceiver`. This 1-D integer Tensor should be batch aligned
    with features, predictions and labels and each element in it is an index in
    the raw input tensor to identify which input each feature / prediction /
    label came from. See
    `eval_saved_model/example_trainers/fake_multi_examples_per_input_estimator.py`
    for an example.
*   Added support for metrics with string `value_op`s.
*   Added support for metrics whose `value_op`s return multidimensional arrays.
*   We now support including your serving graph in the EvalSavedModel. You can
    do this by passing a `serving_input_receiver_fn` to `export_eval_savedmodel`
    or any of the TFMA Exporters.
*   We now support customizing prediction and label keys for
    post_export_metrics.

## Bug fixes and other changes

*   Depends on `apache-beam[gcp]>=2.8,<3`.
*   Depends on `tensorflow-transform>=0.11,<1`.
*   Requires pre-installed TensorFlow >=1.11,<2.
*   Factor our utility functions for adding sliceable "meta-features" to FPL.
*   Added public API docs
*   Add an extractor to add sliceable "meta-features" to FPL.
*   Potentially improved performance by fanning out large slices.
*   Add support for assets_extra in `tfma.exporter.FinalExporter`.
*   Add a light-weight library that includes only the export-related modules for
    TFMA for use in your Trainer. See docstring in
    `tensorflow_model_analysis/export_only/__init__.py` for usage.
*   Update `EvalInputReceiver` so the TFMA collections written to the graph only
    contain the results of the last call if multiple calls to
    `EvalInputReceiver` are made.
*   We now finalize the graph after it's loaded and post-export metrics are
    added, potentially improving performance.
*   Fix a bug in post-export PrecisionRecallAtK where labels with only 1
    dimension were not correctly handled.
*   Fix an issue where we were not correctly wrapping SparseTensors for
    `features` and `labels` in `tf.identity`, which could cause TFMA to
    encounter TensorFlow issue #17568 if there were control dependencies on
    these `features` or `labels`.
*   We now correctly preserve `dtypes` when splitting and concatenating
    SparseTensors internally. The failure to do so previously could result in
    unexpectedly large memory usage if string values were involved due to the
    inefficient pickling of NumPy string arrays with a large number of elements.

## Breaking changes

*   Requires pre-installed TensorFlow >=1.11,<2.
*   We now require that `EvalInputReceiver`, `export_eval_savedmodel`,
    `make_export_strategy`, `make_final_exporter`, `FinalExporter` and
    `LatestExporter` be called with keyword arguments only.
*   Removed `check_metric_compatibility` from `EvalSavedModel`.
*   We now enforce that the `receiver_tensors` dictionary for
    `EvalInputReceiver` contains exactly one key named `examples`.
*   Post-export metrics have now been moved up one level to
    `tfma.post_export_metrics`. They should now be accessed via
    `tfma.post_export_metrics.auc` instead of
    `tfma.post_export_metrics.post_export_metrics.auc` as they were before.
*   Separated extraction from evaluation. `EvaluateAndWriteResults` is now
    called `ExtractEvaluateAndWriteResults`.
*   Added `EvalSharedModel` type to encapsulate `model_path` and
    `add_metrics_callbacks` along with a handle to a shared model instance.

## Deprecations

# Release 0.9.2

## Major Features and Improvements

*   Improved performance especially when slicing across many features and/or
    feature values.

## Bug fixes and other changes

*   Depends on `tensorflow-transform>=0.9,<1`.
*   Requires pre-installed TensorFlow >=1.9,<2.

## Breaking changes

## Deprecations

# Release 0.9.1

## Major Features and Improvements

## Bug fixes and other changes

*   Depends on `apache-beam[gcp]>=2.6,<3`.
*   Updated ExampleCount to use the batch dimension as the example count. It
    also now tries a few fallbacks if none of the standard keys are found in the
    predictions dictionary: the first key in sorted order in the predictions
    dictionary, or failing that, the first key in sorted order in the labels
    dictionary, or failing that, it defaults to zero.
*   Fix bug where we were mutating an element in a DoFn - this is prohibited in
    the Beam model and can cause subtle bugs.
*   Fix bug where we were creating a separate Shared handle for each stage in
    Evaluate, resulting in no sharing of the model across stages.

## Breaking changes

*   Requires pre-installed TensorFlow >=1.10,<2.

## Deprecations

# Release 0.9.0

## Major Features and Improvements

*   Add a TFMA unit test library for unit testing your the exported model and
    associated metrics computations.
*   Add `tfma.export.make_export_strategy` which is analogous to
    `tf.contrib.learn.make_export_strategy`.
*   Add `tfma.exporter.FinalExporter` and `tfma.exporter.LatestExporter` which
    are analogous to `tf.estimator.FinalExporter` and
    `tf.estimator.LastExporter`.
*   Add `tfma.export.build_parsing_eval_input_receiver_fn` which is analogous to
    `tf.estimator.export.build_parsing_serving_input_receiver_fn`.
*   Add integration testing for DNN-based estimators.
*   Add new post export metrics:
    *   AUC (`tfma.post_export_metrics.post_export_metrics.auc`)
    *   Precision/Recall at K
        (`tfma.post_export_metrics.post_export_metrics.precision_recall_at_k`)
    *   Confusion matrix at thresholds
        (`tfma.post_export_metrics.post_export_metrics.confusion_matrix_at_thresholds`)

## Bug fixes and other changes

*   Peak memory usage for large DataFlow jobs should be lower with a fix in when
    we compact batches of metrics during the combine phase of metrics
    computation.
*   Remove batch size override in `chicago_taxi` example.
*   Added dependency on `protobuf>=3.6.0<4` for protocol buffers.
*   Updated SparseTensor code to work with SparseTensors of any dimension.
    Previously on SparseTensors with dimension 2 (batch_size x values) were
    supported in the features dictionary.
*   Updated code to work with SparseTensors and dense Tensors of variable
    lengths across batches.

## Breaking changes

*   EvalSavedModels produced by TFMA 0.6.0 will not be compatible with later
    versions due to the following changes:
    *   EvalSavedModels are now written out with a custom "eval_saved_model"
        tag, as opposed to the "serving" tag before.
    *   EvalSavedModels now include version metadata about the TFMA version that
        they were exported with.
*   Metrics and plot outputs now are converted into proto and serialized.
    Metrics and plots produced by TFMA 0.6.0 will not be compatible with later
    versions.
*   Requires pre-installed TensorFlow >=1.9,<2.
*   TFMA now uses the TensorFlow Estimator functionality for exporting models of
    different modes behind the scenes. There are no user-facing changes
    API-wise, but EvalSavedModels produced by earlier versions of TFMA will not
    be compatible with this version of TFMA.
*   tf.contrib.learn Estimators are no longer supported by TFMA. Only
    tf.estimator Estimators are supported.
*   Metrics and plot outputs now include version metadata about the TFMA version
    that they were exported with. Metrics and plots produced by earlier versions
    of TFMA will not be compatible with this version of TFMA.

## Deprecations

# Release 0.6.0

*   Initial release of TensorFlow Model Analysis.
