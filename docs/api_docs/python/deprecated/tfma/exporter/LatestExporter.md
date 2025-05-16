<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.exporter.LatestExporter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="export"/>
</div>

# tfma.exporter.LatestExporter

## Class `LatestExporter`

Defined in
[`eval_saved_model/exporter.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/eval_saved_model/exporter.py).

<!-- Placeholder for "Used in" -->

This class regularly exports the EvalSavedModel.

In addition to exporting, this class also garbage collects stale exports.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    *args,
    **kwargs
)
```

Wrapped function.

## Properties

<h3 id="name"><code>name</code></h3>

Directory name.

A directory name under the export base directory where exports of this type are
written. Should not be `None` nor empty.

## Methods

<h3 id="export"><code>export</code></h3>

```python
export(
    estimator,
    export_path,
    checkpoint_path,
    eval_result,
    is_the_final_export
)
```

Exports the given `Estimator` to a specific format.

#### Args:

*   <b>`estimator`</b>: the `Estimator` to export.
*   <b>`export_path`</b>: A string containing a directory where to write the
    export.
*   <b>`checkpoint_path`</b>: The checkpoint path to export.
*   <b>`eval_result`</b>: The output of `Estimator.evaluate` on this checkpoint.
*   <b>`is_the_final_export`</b>: This boolean is True when this is an export in
    the end of training. It is False for the intermediate exports during the
    training. When passing `Exporter` to `tf.estimator.train_and_evaluate`
    `is_the_final_export` is always False if `TrainSpec.max_steps` is `None`.

#### Returns:

The string path to the exported directory or `None` if export is skipped.
