# Current version (not yet released; still in development)

## Major Features and Improvements

*   Added support for customizing the pipeline (via extractors, evaluators, and
    writers). See [architecture](g3doc/architecture.md) for more details.
*   Added support for excluding the default metrics from the saved model graph
    during evaluation.
*   Added a mechanism for performing evaluations via post_export_metrics without
    access to a Tensorflow EvalSavedModel.

## Bug fixes and other changes

*   We now store feature and label Tensor information in SignatureDef inputs
    instead of Collections in anticipation of Collections being deprecated in TF
    2.0.
*   Add support for fractional labels in AUC, AUPRC and confusion matrix at
    thresholds. Previously the labels were being passed directly to TensorFlow,
    which would cast them to `bool`, which meant that all non-zero labels were
    treated as positive examples. Now we treat a fractional label `l` in `[0,
    1]` as two examples, a positive example with weight `l` and a negative
    example with weight `1 - l`.

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
