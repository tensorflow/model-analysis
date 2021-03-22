<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug fixes and other Changes

*   For lift metrics, support negative values in the Fairness Indicator UI bar chart.
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

## Breaking Changes

## Deprecations

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
