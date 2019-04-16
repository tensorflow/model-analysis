<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.evaluators.MetricsAndPlotsEvaluator" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.evaluators.MetricsAndPlotsEvaluator

```python
tfma.evaluators.MetricsAndPlotsEvaluator(
    eval_shared_model,
    desired_batch_size=None,
    metrics_key=constants.METRICS_KEY,
    plots_key=constants.PLOTS_KEY,
    run_after=slice_key_extractor.SLICE_KEY_EXTRACTOR_STAGE_NAME,
    num_bootstrap_samples=1
)
```

Defined in
[`evaluators/metrics_and_plots_evaluator.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/evaluators/metrics_and_plots_evaluator.py).

<!-- Placeholder for "Used in" -->

Creates an Evaluator for evaluating metrics and plots.

#### Args:

*   <b>`eval_shared_model`</b>: Shared model parameters for EvalSavedModel.
*   <b>`desired_batch_size`</b>: Optional batch size for batching in Aggregate.
*   <b>`metrics_key`</b>: Name to use for metrics key in Evaluation output.
*   <b>`plots_key`</b>: Name to use for plots key in Evaluation output.
*   <b>`run_after`</b>: Extractor to run after (None means before any
    extractors).
*   <b>`num_bootstrap_samples`</b>: Number of bootstrap samples to draw. If more
    than 1, confidence intervals will be computed for metrics. Suggested value
    is at least 20.

#### Returns:

Evaluator for evaluating metrics and plots. The output will be stored under
'metrics' and 'plots' keys.
