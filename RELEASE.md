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
    pip install -i https://pypi-nightly.tensorflow.org/simple tensorflow-model-analysis
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of TFMA available on PyPI by running the
    command `pip install tensorflow-model-analysis` .

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
