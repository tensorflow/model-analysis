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

