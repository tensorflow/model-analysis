<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.default_eval_shared_model" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.default_eval_shared_model

```python
tfma.default_eval_shared_model(
    eval_saved_model_path,
    add_metrics_callbacks=None,
    include_default_metrics=True,
    example_weight_key=None,
    additional_fetches=None
)
```

Defined in
[`api/model_eval_lib.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/api/model_eval_lib.py).

<!-- Placeholder for "Used in" -->

Returns default EvalSharedModel.

#### Args:

*   <b>`eval_saved_model_path`</b>: Path to EvalSavedModel.
*   <b>`add_metrics_callbacks`</b>: Optional list of callbacks for adding
    additional metrics to the graph (see EvalSharedModel for more information on
    how to configure additional metrics). Metrics for example counts and example
    weight will be added automatically.
*   <b>`include_default_metrics`</b>: True to include the default metrics that
    are part of the saved model graph during evaluation.
*   <b>`example_weight_key`</b>: Deprecated.
*   <b>`additional_fetches`</b>: Prefixes of additional tensors stored in
    signature_def.inputs that should be fetched at prediction time. The
    "features" and "labels" tensors are handled automatically and should not be
    included.
