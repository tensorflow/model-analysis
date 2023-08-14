# Version 0.45.0

## Major Features and Improvements

*   Add F1, False positive rate, and Accuracy into the confusion matrix plot.
*   Add support for setting top_k and class_id at the same time for confusion
    matrix metrics.
*   Add the false positive for semantic segmentation metrics.
*   Add Mean Metric (experimental) which calculates the mean of any feature. *.
    Adds support of `output_keypath` to ModelSignatureDoFn to explicitly set a
    chain of output keys in the multi-level dict (extracts). Adds output_keypath
    to common prediction extractors.

## Bug fixes and other Changes

*   Fix the bug that SetMatchRecall is always 1 when top_k is set.
*   Depends on `pyarrow>=10,<11`.
*   Depends on `apache-beam>=2.47,<3`.
*   Depends on `numpy>=1.23.0`.
*   Depends on `tensorflow>=2.13.0,<3`.
*   Add 'tfma_eval' model_type in model_specs as the identifier for
    eval_saved_model, allowing signature='eval' to now be used with other model
    types.
*   Add "materialized_prediction" model type to allow users bypassing model
    inference explicitly.

## Breaking Changes

*   Depend on PIL for image related metrics.
*   Separate extract_key from signature names in `ModelSignaturesDoFn`.

## Deprecations

*   N/A

