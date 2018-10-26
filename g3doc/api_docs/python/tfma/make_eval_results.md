<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.make_eval_results" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.make_eval_results

``` python
tfma.make_eval_results(
    results,
    mode
)
```

Run model analysis for a single model on multiple data sets.

#### Args:

* <b>`results`</b>: A list of TFMA evaluation results.
* <b>`mode`</b>: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
  tfma.MODEL_CENTRIC_MODE are supported.


#### Returns:

An EvalResults containing all evaluation results. This can be used to
construct a time series view.