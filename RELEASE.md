# Current version (not yet released; still in development)

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

## Bug fixes and other changes

*   Fixed error in `tfma-multi-class-confusion-matrix-at-thresholds` with
    default classNames value.
*   Fairness Indicators: compute ratio metrics with safe division, remove
    "post_export_metrics" from metric names. Fairness Indicators UI now displays
    slices in alphabetic order.

## Breaking changes

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
    with latest master, `tfx-bsl` must also be latest master.
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
*   Added support for blacklisting specified features from fetches.
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
*   Separated extraction from evaluation. `EvaluteAndWriteResults` is now called
    `ExtractEvaluateAndWriteResults`.
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
