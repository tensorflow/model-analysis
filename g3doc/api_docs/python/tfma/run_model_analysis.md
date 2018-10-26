<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.run_model_analysis" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.run_model_analysis

``` python
tfma.run_model_analysis(
    model_location,
    data_location,
    file_format='tfrecords',
    slice_spec=None,
    example_weight_key=None,
    add_metrics_callbacks=None,
    output_path=None,
    extractors=None
)
```

Runs TensorFlow model analysis.

It runs a Beam pipeline to compute the slicing metrics exported in TensorFlow
Eval SavedModel and returns the results.

This is a simplified API for users who want to quickly get something running
locally. Users who wish to create their own Beam pipelines can use the
Evaluate PTransform instead.

#### Args:

* <b>`model_location`</b>: The location of the exported eval saved model.
* <b>`data_location`</b>: The location of the data files.
* <b>`file_format`</b>: The file format of the data, can be either 'text' or
    'tfrecords' for now. By default, 'tfrecords' will be used.
* <b>`slice_spec`</b>: A list of tfma.SingleSliceSpec. Each spec represents a way to
    slice the data.
    Example usages:
    - tfma.SingleSiceSpec(): no slice, metrics are computed on overall data.
    - tfma.SingleSiceSpec(columns=['country']): slice based on features in
      column "country". We might get metrics for slice "country:us",
      "country:jp", and etc in results.
    - tfma.SingleSiceSpec(features=[('country', 'us')]): metrics are computed
      on slice "country:us".
    If None, defaults to the overall slice.
* <b>`example_weight_key`</b>: The key of the example weight column. If None, weight
    will be 1 for each example.
* <b>`add_metrics_callbacks`</b>: Optional list of callbacks for adding additional
    metrics to the graph. The names of the metrics added by the callbacks
    should not conflict with existing metrics, or metrics added by other
    callbacks. See docstring for Evaluate in api/impl/evaluate.py for more
    details.
* <b>`output_path`</b>: The directory to output metrics and results to. If None, we use
    a temporary directory.
* <b>`extractors`</b>: An optional list of PTransforms to run before slicing the data.


#### Returns:

An EvalResult that can be used with the TFMA visualization functions.


#### Raises:

* <b>`ValueError`</b>: If the file_format is unknown to us.