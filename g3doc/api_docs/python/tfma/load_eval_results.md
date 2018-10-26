<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.load_eval_results" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.load_eval_results

``` python
tfma.load_eval_results(
    output_paths,
    mode
)
```

Run model analysis for a single model on multiple data sets.

#### Args:

* <b>`output_paths`</b>: A list of output paths of completed tfma runs.
* <b>`mode`</b>: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
  tfma.MODEL_CENTRIC_MODE are supported.


#### Returns:

An EvalResults containing the evaluation results serialized at output_paths.
This can be used to construct a time series view.