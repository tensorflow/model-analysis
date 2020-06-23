# Version 0.22.0

## Major Features and Improvements

*   Added analyze_raw_data(), an API for evaluating TFMA metrics on Pandas
    DataFrames.

## Bug fixes and other changes

*   Previously metrics would only be computed for combinations of keys that
    produced different metric values (e.g. `ExampleCount` will be the same for
    all models, outputs, classes, etc, so only one metric key was used). Now a
    metric key will be returned for each combination associated with the
    `MetricSpec` definition even if the values will be the same. Support for
    model independent metrics has also been removed. This means by default
    multiple ExampleCount metrics will be created when multiple models are used
    (one per model).
*   Fixed issue with label_key and prediction_key settings not working with TF
    based metrics.
*   Fairness Indicators UI
    *   Thresholds are now sorted in ascending order.
    *   Barchart can now be sorted by either slice or eval.
*   Added support for slicing on any value extracted from the inputs (e.g. raw
    labels).
*   Added support for filtering extracts based on sub-keys.
*   Added beam counters to track the feature slices being used for evaluation.
*   Adding KeyError when analyze_raw_data is run without a valid label_key or
    prediction_key within the provided Pandas DataFrame.
*   Added documentation for `tfma.analyze_raw_data`, `tfma.view.SlicedMetrics`,
    and `tfma.view.SlicedPlots`.
*   Unchecked Metric thresholds now block the model validation.
*   Added support for per slice threshold settings.
*   Added support for sharding metrics and plots outputs.
*   Updated load_eval_result to support filtering plots by model name. Added
    support for loading multiple models at same output path using
    load_eval_results.
*   Fix typo in jupyter widgets breaking TimeSeriesView and PlotViewer.
*   Add `tfma.slicer.stringify_slice_key()`.
*   Deprecated external use of tfma.slicer.SingleSliceSpec (tfma.SlicingSpec
    should be used instead).
*   Updated tfma.default_eval_shared_model and tfma.default_extractors to better
    support custom model types.
*   Depends on 'tensorflow-metadata>=0.22.2,<0.23'

## Breaking changes

*   Changed to treat CLASSIFY_OUTPUT_SCORES involving 2 values as a multi-class
    classification prediction instead of converting to binary classification.
*   Refactored confidence interval methodology field. The old path under
    `Options.confidence_interval_methodology` is now at
    `Options.confidence_intervals.methodology`.
*   Removed model_load_time_callback from ModelLoader construct_fn (timing is
    now handled by load). Removed access to shared_handle from ModelLoader.

## Deprecations
