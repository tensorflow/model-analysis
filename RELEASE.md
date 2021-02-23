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
