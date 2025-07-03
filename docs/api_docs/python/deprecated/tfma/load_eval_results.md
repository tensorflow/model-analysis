<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.load_eval_results" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.load_eval_results

```python
tfma.load_eval_results(
    output_paths,
    mode
)
```

Defined in
[`api/model_eval_lib.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/api/model_eval_lib.py).

<!-- Placeholder for "Used in" -->

Run model analysis for a single model on multiple data sets.

#### Args:

*   <b>`output_paths`</b>: A list of output paths of completed tfma runs.
*   <b>`mode`</b>: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE
    and tfma.MODEL_CENTRIC_MODE are supported.

#### Returns:

An EvalResults containing the evaluation results serialized at output_paths.
This can be used to construct a time series view.
