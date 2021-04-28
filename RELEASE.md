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

## Deprecations

*   N/A
