<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.default_evaluators" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.default_evaluators

```python
tfma.default_evaluators(
    eval_shared_model,
    desired_batch_size=None,
    num_bootstrap_samples=None
)
```

Defined in
[`api/model_eval_lib.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/api/model_eval_lib.py).

<!-- Placeholder for "Used in" -->

Returns the default evaluators for use in ExtractAndEvaluate.

#### Args:

*   <b>`eval_shared_model`</b>: Shared model parameters for EvalSavedModel.
*   <b>`desired_batch_size`</b>: Optional batch size for batching in Aggregate.
*   <b>`num_bootstrap_samples`</b>: Number of bootstrap samples to draw. If more
    than 1, confidence intervals will be computed for metrics. Suggested value
    is at least 20.
