# Version 0.46.0

## Major Features and Improvements

*   Removes the metrics modules from experimental now that it is migrated to
    [py-ml-metrics](https://pypi.org/project/py-ml-metrics/) package.
*   Adds Constituent Flip Rate Metrics: SymmetricFlipRate, NegToNegFlipRate,
    NegToPosFlipRate, PosToNegFlipRate, PosToPosFlipRate.
*   Depend on tensorflow-estimator package explicitly.

## Bug fixes and other Changes

*   Fix the bug about batching unsized numpy arrays.

## Breaking Changes

*   Removes `attrs` requirement.
*   Consolidate Matrix definition for semantic segmentation confusion matrix
    metrics.
*   Provide AggregateFn and interface and default call impl to adapt TFMA
    metrics combiner for in-process call.
*   Move Mean metrics from experimental to metrics.
*   Fix the bug of size estimator failure.
*   Depends on `tensorflow>=2.15.0,<2.16`.
*   Fix the failure in testMeanAttributions.
*   Fix the input type mismatch in metric_specs_tests between bool and None.
*   Fix the failure in the slice test due to beam type hints check.
*   Fix the failure in metric_specs test, all TFMA deps on keras are
    keras 2.
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on
    `apache-beam[gcp]>=2.47.0,<3` for 3.9 and 3.10.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.
*   Update the minimum Bazel version required to build TFMA to 6.1.0
*   Refactors BooleanFlipRates computations to a combiner (flip_counts) and a
    DerivedMetricComputation (flip_rates).

## Deprecations

*   Deprecated python 3.8 support.

