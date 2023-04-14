# Version 0.44.0

## Major Features and Improvements

*   Add BinaryCrossEntropy and CategoricalCrossEntropy.
*   Add MeanAbsolutePercentageError and MeanSquaredLogarithmicError
*   Add SetMatchPrecision and SetMatchRecall

## Bug fixes and other Changes

*   Fix for jupiter notebook
*   Fix element dimension inconsistency when some of the extracts have missing
    key.
*   Add public visibility to the servo beam extractor.
*   Fix for bug where binary_confusion_matrices with different class_weights are
    considered identical and deduplicated.
*   Fixes bug where positive / negative axes labels are reversed in prediction
    distribution plot.
*   Depends on `numpy~=1.22.0`.
*   Modify ExampleCount to not depend on labels and predictions.
*   Add class_id info into sub_key in metric_key for object detection confusion
    matrix metrics.
*   Add class_id info into sub_key in plot_key for object detection confusion
    matrix plot.
*   Fix a bug that auto_pivot dropped nan when deciding which columns are
    multivalent for pivoting.
*   Depends on `tensorflow>=2.12.0,<2.13`.
*   Depends on `protobuf>=3.20.3,<5`.
*   Depends on `tfx-bsl>=1.13.0,<1.14.0`.
*   Depends on `tensorflow-metadata>=1.13.1,<1.14.0`.

## Breaking Changes

*   N/A

## Deprecations

Deprecated python3.7 support.

