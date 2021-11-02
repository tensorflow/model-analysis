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

