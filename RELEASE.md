# Version 0.27.0

## Major Features and Improvements

*   Created tfma.StandardExtracts with helper methods for common keys.
*   Updated StandardMetricInputs to extend from the tfma.StandardExtracts.
*   Created set of StandardMetricInputsPreprocessors for filtering extracts.
*   Introduced a `padding_options` config to ModelSpec to configure whether
    and how to pad the prediction and label tensors expected by the model's
    metrics.

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
