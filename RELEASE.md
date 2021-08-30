# Version 0.34.0

## Major Features and Improvements

*   Added `SparseTensorValue` and `RaggedTensorValue` types for storing
    in-memory versions of sparse and ragged tensor values used in extracts.
    Tensor values used for features, etc should now be based on either
    `np.ndarray`, `SparseTensorValue`, or `RaggedTensorValue` and not
    tf.compat.v1 value types.
*   Add `CIDerivedMetricComputation` metric type.

## Bug fixes and other Changes

*   Fixes bug when computing confidence intervals for
    `binary_confusion_metrics.ConfusionMatrixAtThresholds` (or any other
    structured metric).
*   Fixed bug where example_count post_export_metric is added even if
    include_default_metrics is False.
*   Depends on `apache-beam[gcp]>=2.31,<2.32`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3`.
*   Depends on `tfx-bsl>=1.3.1,<1.4.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A
