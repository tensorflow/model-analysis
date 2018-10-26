<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.multiple_data_analysis" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.multiple_data_analysis

``` python
tfma.multiple_data_analysis(
    model_location,
    data_locations,
    **kwargs
)
```

Run model analysis for a single model on multiple data sets.

#### Args:

* <b>`model_location`</b>: The location of the exported eval saved model.
* <b>`data_locations`</b>: A list of data set locations.
* <b>`**kwargs`</b>: The args used for evaluation. See tfma.run_model_analysis() for
    details.


#### Returns:

A tfma.EvalResults containing all the evaluation results with the same order
as data_locations.