<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.default_extractors" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.default_extractors

```python
tfma.default_extractors(
    eval_shared_model,
    slice_spec=None,
    desired_batch_size=None,
    materialize=True
)
```

Defined in
[`api/model_eval_lib.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/api/model_eval_lib.py).

<!-- Placeholder for "Used in" -->

Returns the default extractors for use in ExtractAndEvaluate.

#### Args:

*   <b>`eval_shared_model`</b>: Shared model parameters for EvalSavedModel.
*   <b>`slice_spec`</b>: Optional list of SingleSliceSpec specifying the slices
    to slice the data into. If None, defaults to the overall slice.
*   <b>`desired_batch_size`</b>: Optional batch size for batching in Aggregate.
*   <b>`materialize`</b>: True to have extractors create materialized output.
