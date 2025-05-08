<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.run_model_analysis" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.run_model_analysis

```python
tfma.run_model_analysis(
    eval_shared_model,
    data_location,
    file_format='tfrecords',
    slice_spec=None,
    output_path=None,
    extractors=None,
    evaluators=None,
    writers=None,
    write_config=True,
    pipeline_options=None,
    num_bootstrap_samples=1
)
```

Defined in
[`api/model_eval_lib.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/api/model_eval_lib.py).

<!-- Placeholder for "Used in" -->

Runs TensorFlow model analysis.

It runs a Beam pipeline to compute the slicing metrics exported in TensorFlow
Eval SavedModel and returns the results.

This is a simplified API for users who want to quickly get something running
locally. Users who wish to create their own Beam pipelines can use the Evaluate
PTransform instead.

#### Args:

*   <b>`eval_shared_model`</b>: Shared model parameters for EvalSavedModel
    including any additional metrics (see EvalSharedModel for more information
    on how to configure additional metrics).
*   <b>`data_location`</b>: The location of the data files.
*   <b>`file_format`</b>: The file format of the data, can be either 'text' or
    'tfrecords' for now. By default, 'tfrecords' will be used.
*   <b>`slice_spec`</b>: A list of tfma.slicer.SingleSliceSpec. Each spec
    represents a way to slice the data. If None, defaults to the overall slice.
    Example usages: # TODO(xinzha): add more use cases once they are supported
    in frontend.
    -   tfma.SingleSiceSpec(): no slice, metrics are computed on overall data.
    -   tfma.SingleSiceSpec(columns=['country']): slice based on features in
        column "country". We might get metrics for slice "country:us",
        "country:jp", and etc in results.
    -   tfma.SingleSiceSpec(features=[('country', 'us')]): metrics are computed
        on slice "country:us".
*   <b>`output_path`</b>: The directory to output metrics and results to. If
    None, we use a temporary directory.
*   <b>`extractors`</b>: Optional list of Extractors to apply to Extracts.
    Typically these will be added by calling the default_extractors function. If
    no extractors are provided, default_extractors (non-materialized) will be
    used.
*   <b>`evaluators`</b>: Optional list of Evaluators for evaluating Extracts.
    Typically these will be added by calling the default_evaluators function. If
    no evaluators are provided, default_evaluators will be used.
*   <b>`writers`</b>: Optional list of Writers for writing Evaluation output.
    Typically these will be added by calling the default_writers function. If no
    writers are provided, default_writers will be used.
*   <b>`write_config`</b>: True to write the config along with the results.
*   <b>`pipeline_options`</b>: Optional arguments to run the Pipeline, for
    instance whether to run directly.
*   <b>`num_bootstrap_samples`</b>: Optional, set to at least 20 in order to
    calculate metrics with confidence intervals.

#### Returns:

An EvalResult that can be used with the TFMA visualization functions.

#### Raises:

*   <b>`ValueError`</b>: If the file_format is unknown to us.
