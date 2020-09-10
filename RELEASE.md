# Version 0.24.0

## Major Features and Improvements

*   Use TFXIO and batched extractors by default in TFMA.

## Bug fixes and other changes

*   Updated the type hint of FilterOutSlices.
*   Fix issue with precision@k and recall@k giving incorrect values when
    negative thresholds are used (i.e. keras defaults).
*   Fix issue with MultiClassConfusionMatrixPlot being overridden by
    MultiClassConfusionMatrix metrics.
*   Made the Fairness Indicators UI thresholds drop down list sorted.
*   Fix the bug that Sort menu is not hidden when there is no model comparison.
*   Depends on `absl-py>=0.9,<0.11`.
*   Depends on `ipython>=7,<8`.
*   Depends on `pandas>=1.0,<2`.
*   Depends on `protobuf>=3.9.2,<4`.
*   Depends on `tensorflow-metadata>=0.24.0,<0.25.0`.
*   Depends on `tfx-bsl>=0.24.0,<0.25.0`.

## Breaking changes

*   Query based metrics evaluations that make use of `MetricsSpecs.query_key`
    are now passed `tfma.Extracts` with leaf values that are of type
    `np.ndarray` containing an additional dimension representing the values
    matched by the query (e.g. if the labels and predictions were previously 1D
    arrays, they will now be 2D arrays where the first dimension's size is equal
    to the number of examples matching the query key). Previously a list of
    `tfma.Extracts` was passed instead. This allows user's to now add custom
    metrics based on `tf.keras.metrics.Metric` as well as `tf.metrics.Metric`
    (any previous customizations based on `tf.metrics.Metric` will need to be
    updated). As part of this change the `tfma.metrics.NDCG`,
    `tfma.metrics.MinValuePosition`, and `tfma.metrics.QueryStatistics` have
    been updated.
*   Renamed `ConfusionMatrixMetric.compute` to `ConfusionMatrixMetric.result`
    for consistency with other APIs.

## Deprecations

*   Deprecating Py3.5 support.
