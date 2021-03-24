# Version 0.29.0

## Major Features and Improvements

*   Added support for output aggregation.

## Bug fixes and other Changes

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
*   Fix support for loading the UI outside of a notebook.
*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `tensorflow-metadata>=0.29.0,<0.30.0`.
*   Depends on `tfx-bsl>=0.29.0,<0.30.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A
