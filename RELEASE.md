# Release 0.9.2

## Major Features and Improvements

## Bug fixes and other changes

*   Depends on `tensorflow-transform>=0.9,<1`.
*   Requires pre-installed TensorFlow >=1.9,<2.

## Breaking changes

## Deprecations

# Release 0.9.1

## Major Features and Improvements

## Bug fixes and other changes

*   Depends on `apache-beam[gcp]>=2.6,<3`.
*   Requires pre-installed TensorFlow >=1.10,<2.
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

 * Initial release of TensorFlow Model Analysis.
