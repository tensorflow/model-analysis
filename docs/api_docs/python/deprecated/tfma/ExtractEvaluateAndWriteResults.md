<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.ExtractEvaluateAndWriteResults" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.ExtractEvaluateAndWriteResults

```python
tfma.ExtractEvaluateAndWriteResults(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

PTransform for performing extraction, evaluation, and writing results.

Users who want to construct their own Beam pipelines instead of using the
lightweight run_model_analysis functions should use this PTransform.

Example usage: eval_shared_model = tfma.default_eval_shared_model(
eval_saved_model_path=model_location, add_metrics_callbacks=[...]) with
beam.Pipeline(runner=...) as p: _ = (p | 'ReadData' >>
beam.io.ReadFromTFRecord(data_location) | 'ExtractEvaluateAndWriteResults' >>
tfma.ExtractEvaluateAndWriteResults( eval_shared_model=eval_shared_model,
output_path=output_path, display_only_data_location=data_location,
slice_spec=slice_spec, ...)) result =
tfma.load_eval_result(output_path=output_path)
tfma.view.render_slicing_metrics(result)

Note that the exact serialization format is an internal implementation detail
and subject to change. Users should only use the TFMA functions to write and
read the results.

#### Args:

*   <b>`examples`</b>: PCollection of input examples. Can be any format the
    model accepts (e.g. string containing CSV row, TensorFlow.Example, etc).
*   <b>`eval_shared_model`</b>: Shared model parameters for EvalSavedModel
    including any additional metrics (see EvalSharedModel for more information
    on how to configure additional metrics).
*   <b>`output_path`</b>: Path to output metrics and plots results.
*   <b>`display_only_data_location`</b>: Optional path indicating where the
    examples were read from. This is used only for display purposes - data will
    not actually be read from this path.
*   <b>`slice_spec`</b>: Optional list of SingleSliceSpec specifying the slices
    to slice the data into. If None, defaults to the overall slice.
*   <b>`desired_batch_size`</b>: Optional batch size for batching in Predict and
    Aggregate.
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
*   <b>`num_bootstrap_samples`</b>: Optional, set to at least 20 in order to
    calculate metrics with confidence intervals.

#### Raises:

*   <b>`ValueError`</b>: If matching Extractor not found for an Evaluator.

#### Returns:

PDone.
