<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.multiple_model_analysis" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.multiple_model_analysis

```python
tfma.multiple_model_analysis(
    model_locations,
    data_location,
    **kwargs
)
```

Defined in
[`api/model_eval_lib.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/api/model_eval_lib.py).

<!-- Placeholder for "Used in" -->

Run model analysis for multiple models on the same data set.

#### Args:

*   <b>`model_locations`</b>: A list of paths to the export eval saved model.
*   <b>`data_location`</b>: The location of the data files.
*   <b>`**kwargs`</b>: The args used for evaluation. See
    tfma.run_model_analysis() for details.

#### Returns:

A tfma.EvalResults containing all the evaluation results with the same order as
model_locations.
