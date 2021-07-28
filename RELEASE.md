# Version 0.33.0

## Major Features and Improvements

*   Provided functionality for `slice_keys_sql` config. It's not available under
    Windows.

## Bug fixes and other Changes

*   Improve rendering of HTML stubs for TFMA and Fairness Indicators UI.
*   Update README for JupyterLab 3
*   Provide implementation of ExactMatch metric.
*   Jackknife CI method now works with cross-slice metrics.
*   Depends on `apache-beam[gcp]>=2.31,<3`.
*   Depends on `tensorflow-metadata>=1.2.0,<1.3.0`.
*   Depends on `tfx-bsl>=1.2.0,<1.3.0`.

## Breaking Changes

*   The binary_confusion_matrices metric formerly returned confusion matrix
    counts (i.e number of {true,false} {positives,negatives}) and optionally a
    set of representative examples in a single object. Now, this metric class
    generates two separate metrics values when examples are configured: one
    containing just the counts, and the other just examples. This should only
    affect users who created a custom derived metric that used
    binary_confusion_matrices metric as an input.

## Deprecations

*   N/A
