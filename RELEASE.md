# Current version (not yet released; still in development)

## Major Features and Improvements
 * Add `tfma.export.make_export_strategy` which is analogous to
   `tf.contrib.learn.make_export_strategy`.
 * Add `tfma.exporter.FinalExporter` and `tfma.exporter.LatestExporter` which
   are analogous to `tf.estimator.FinalExporter` and
   `tf.estimator.LastExporter`.

## Bug fixes and other changes
 * Peak memory usage for large DataFlow jobs should be lower with a fix in when
   we compact batches of metrics during the combine phase of metrics
   computation.
 * Remove batch size override in `chicago_taxi` example.

## Breaking changes
 * EvalSavedModels produced by TFMA 0.6.0 will not be compatible with TFMA 0.7.0
   due to the following changes:
   * EvalSavedModels are now written out with a custom "eval_saved_model" tag,
     as opposed to the "serving" tag before.
   * EvalSavedModels now include version metadata about the TFMA version that
     they were exported with.
 * Metrics and plot outputs now include version metadata about the TFMA version
   that they were exported with. Metrics and plots produced by TFMA 0.6.0 will
   not be compatible with TFMA 0.7.0.

## Deprecations

# Release 0.6.0
 * Initial release of TensorFlow Model Analysis.
