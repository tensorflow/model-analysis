# Version 0.36.0

## Major Features and Improvements

*   Replaced keras metrics with TFMA implementations. To use a keras metric in a
    `tfma.MetricConfig` you must now specify a module (i.e. `tf.keras.metrics`).
*   Added FixedSizeSample metric which can be used to extract a random,
    per-slice, fixed-sized sample of values for a user-configured feature key.

## Bug fixes and other Changes

*   Updated QueryStatistics to support weighted examples.
*   Depends on `apache-beam[gcp]>=2.34,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `tfx-bsl>=1.5.0,<1.6.0`.
*   Depends on `tensorflow-metadata>=1.5.0,<1.6.0`.

## Breaking Changes

*   Removes register_metric from public API, as it is not intended to be public
    facing. To use a custom metric, provide the module name in which the
    metric is defined in the MetricConfig message, instead.

## Deprecations

