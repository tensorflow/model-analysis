# Version 0.23.0

## Major Features and Improvements

*   Changed default confidence interval method from POISSON_BOOTSTRAP to
    JACKKNIFE. This should significantly improve confidence interval evaluation
    performance by as much as 10x in runtime and CPU resource usage.
*   Added support for additional confusion matrix metrics (FDR, FOR, PT, TS, BA,
    F1 score, MCC, FM, Informedness, Markedness, etc). See
    https://en.wikipedia.org/wiki/Confusion_matrix for full list of metrics now
    supported.
*   Change the number of partitions used by the JACKKNIFE confidence interval
    methodology from 100 to 20. This will reduce the quality of the confidence
    intervals but support computing confidence intervals on slices with fewer
    examples.
*   Added `tfma.metrics.MultiClassConfusionMatrixAtThresholds`.
*   Refactoring code to compute `tfma.metrics.MultiClassConfusionMatrixPlot`
    using derived computations.

## Bug fixes and other changes

*   Added support for labels passed as SparseTensorValues.
*   Stopped requiring `avro-python3`.
*   Fix NoneType error when passing BinarizeOptions to
    tfma.metrics.default_multi_class_classification_specs.
*   Fix issue with custom metrics contained in modules ending in
    tf.keras.metric.
*   Changed the BoundedValue.value to be the unsampled metric value rather than
    the sample average.
*   Add `EvalResult.get_metric_names()`.
*   Added errors for missing slices during metrics validation.
*   Added support for customizing confusion matrix based metrics in keras.
*   Made BatchedInputExtractor externally visible.
*   Updated tfma.load_eval_results API to return empty results instead of
    throwing an error when evaluation results are missing for a model_name.
*   Fixed an issue in Fairness Indicators UI where omitted slices error message
    was being displayed even if no slice was omitted.
*   Fix issue with slice_spec.is_slice_applicable not working for float, int,
    etc types that are encoded as strings.
*   Wrap long strings in table cells in Fairness Indicators UI
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `scipy>=1.4.1,<2`
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
*   Depends on `tensorflow-metadata>=0.23,<0.24`.
*   Depends on `tfx-bsl>=0.23,<0.24`.

## Breaking changes

*   Rename EvalResult.get_slices() to EvalResult.get_slice_names().

## Deprecations

*   N/A
